"""
Does LIGHTING explain why the ANN misses closed eyes on the webcam?  (question from the supervisor)

Our models already standardise every image (mean 0, std 1) as their FIRST layer, which removes a global
brightness / contrast change. This script tests what is left, with numbers, in three parts:

  Part 1  SENSITIVITY (FL3D validation images, same split as training): change the lighting of known-good
          images (darker, brighter, blue cast, side light) and the framing (shift, zoom) and see which
          model collapses.  If the ANN stays accurate under dark/bright but collapses under shift/zoom,
          lighting is NOT the main cause - framing is.
  Part 2  PRE-PROCESSING FIX (your webcam images, test-time only): apply an extra step BEFORE the model
          (per-channel gray-world, CLAHE contrast, auto-gamma) and see if the ANN's microsleep recall
          recovers.  No retraining needed for this first look.
  Part 3  DIAGNOSTIC: average probability the ANN / CNN gives to "microsleep" on your microsleep images
          (close to alert = a threshold problem, near zero = the model sees nothing).

    python src/lighting_experiment.py --fl3d dataset_prepared --images my_images ^
        --models models/img_ann.pt models/img_cnn.pt

Outputs: reports/lighting/lighting_report.md and lighting_results.csv.  Nothing in the repo is modified.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgmodels as im  # noqa: E402
from train_img import split_indices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ image operations (uint8, N x 3 x H x W, RGB)
def _clip(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def op_none(X):
    return X


def op_bright(f):
    return lambda X: _clip(X.astype(np.float32) * f)


def op_blue_cast(X):
    k = np.array([0.8, 1.0, 1.25], np.float32)[None, :, None, None]       # R down, B up
    return _clip(X.astype(np.float32) * k)


def op_side_light(X):
    w = X.shape[3]
    ramp = np.linspace(0.45, 1.55, w, dtype=np.float32)[None, None, None, :]   # dark left -> bright right
    return _clip(X.astype(np.float32) * ramp)


def op_shift(px):
    def f(X):
        out = np.roll(X, shift=(px, px), axis=(2, 3))
        return out
    return f


def op_zoom(factor):
    """factor<1: crop the centre and enlarge (face looks bigger); factor>1: shrink the face with padding."""
    import cv2

    def f(X):
        n, c, h, w = X.shape
        out = np.empty_like(X)
        for i in range(n):
            img = np.ascontiguousarray(X[i].transpose(1, 2, 0))
            if factor < 1:
                m = int(round(h * (1 - factor) / 2))
                img = cv2.resize(img[m:h - m, m:w - m], (w, h), interpolation=cv2.INTER_AREA)
            else:
                s = int(round(h / factor))
                small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
                canvas = np.full_like(img, int(img.mean()))
                o = (h - s) // 2
                canvas[o:o + s, o:o + s] = small
                img = canvas
            out[i] = img.transpose(2, 0, 1)
        return out
    return f


def op_grayworld(X):
    """Per-channel gain so that R, G and B have the same mean (removes colour casts; the model's own
    standardisation uses ONE mean/std for all channels, so it cannot remove a blue cast)."""
    x = X.astype(np.float32)
    m = x.mean(axis=(2, 3), keepdims=True)
    return _clip(x * (m.mean(axis=1, keepdims=True) / np.maximum(m, 1e-3)))


def op_clahe(X):
    import cv2
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    out = np.empty_like(X)
    for i in range(len(X)):
        bgr = np.ascontiguousarray(X[i][::-1].transpose(1, 2, 0))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        out[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[:, :, ::-1].transpose(2, 0, 1)
    return out


def op_autogamma(X):
    """Gamma so that the mean luminance of each image becomes 0.5."""
    x = X.astype(np.float32) / 255.0
    lum = np.clip(x.mean(axis=(1, 2, 3), keepdims=True), 0.02, 0.98)
    g = np.log(0.5) / np.log(lum)
    return _clip((x ** g) * 255.0)


def op_gw_clahe(X):
    return op_clahe(op_grayworld(X))


# ------------------------------------------------------------------ evaluation
def evaluate(model, X, y, device, op, n_classes):
    p = im.predict_proba(model, op(X), device)
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    rec = []
    for c in range(n_classes):
        m = y == c
        rec.append(float((pred[m] == c).mean()) if m.any() else float("nan"))
    return acc, rec, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fl3d", required=True, help="prepared FL3D folder (dataset_prepared)")
    ap.add_argument("--images", required=True, help="your webcam images (class sub-folders)")
    ap.add_argument("--models", nargs=2, required=True, help="ANN checkpoint then CNN checkpoint")
    ap.add_argument("--seed", type=int, default=0, help="same seed as training (default 0)")
    ap.add_argument("--max-val", type=int, default=3000, help="images of the FL3D val split to use (speed)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "lighting"))
    args = ap.parse_args()

    device = im.get_device()
    models = {}
    for path in args.models:
        m, ck = im.load_checkpoint(path, device)
        models[ck["arch"].upper()] = m
        img, gray, classes = ck["img"], ck["gray"], ck["classes"]
    if "ANN" not in models or "CNN" not in models:
        raise SystemExit("Give one ANN and one CNN checkpoint.")
    micro = classes.index("microsleep") if "microsleep" in classes else 1
    print("classes:", classes)

    print("Loading FL3D ...")
    X, y, cl2, paths = im.load_folder(args.fl3d, img, gray, classes=classes)
    _, va, _ = split_indices(paths, y, 0.2, args.seed)
    rng = np.random.default_rng(0)
    if len(va) > args.max_val:
        va = np.sort(rng.choice(va, args.max_val, replace=False))
    Xv, yv = X[va], y[va]
    del X
    Xw, yw, _, _ = im.load_folder(args.images, img, gray, classes=classes, verbose=False)
    print(f"FL3D val used: {len(Xv)} | webcam images: {len(Xw)} {np.bincount(yw).tolist()}")

    rows = []
    md = ["# Lighting / framing experiment", "",
          f"Classes: {classes}. Recall = per-class recall; `micro` = microsleep (eyes closed).", ""]

    def run(part, title, ops, X_, y_):
        md.extend([f"## {title}", "", "| Condition | ANN acc | ANN micro recall | CNN acc | CNN micro recall |", "|---|---|---|---|---|"])
        for name, op in ops:
            cells = []
            for arch in ("ANN", "CNN"):
                acc, rec, _ = evaluate(models[arch], X_, y_, device, op, len(classes))
                cells += [f"{acc*100:.1f}%", f"{rec[micro]*100:.1f}%"]
                rows.append([part, name, arch, f"{acc:.4f}", f"{rec[micro]:.4f}"])
            md.append(f"| {name} | " + " | ".join(cells) + " |")
            print(f"[{part}] {name:28s} ANN {cells[0]:>7s} / micro {cells[1]:>7s}   CNN {cells[2]:>7s} / micro {cells[3]:>7s}")
        md.append("")

    sens = [("none", op_none), ("darker x0.4", op_bright(0.4)), ("darker x0.7", op_bright(0.7)),
            ("brighter x1.5", op_bright(1.5)), ("blue colour cast", op_blue_cast), ("side light (gradient)", op_side_light),
            ("shift 4 px", op_shift(4)), ("zoom in (crop 85%)", op_zoom(0.85)), ("zoom out (face 85%)", op_zoom(1 / 0.85))]
    run("P1", "Part 1 - sensitivity on FL3D validation images (known-good images, one change at a time)", sens, Xv, yv)

    fixes = [("none (current)", op_none), ("gray-world (per-channel)", op_grayworld), ("CLAHE contrast", op_clahe),
             ("auto-gamma (mean 0.5)", op_autogamma), ("gray-world + CLAHE", op_gw_clahe)]
    run("P2", "Part 2 - extra pre-processing before the model, on YOUR webcam images", fixes, Xw, yw)

    run("P2b", "Part 2b - framing change on YOUR webcam images", [("none", op_none), ("zoom in (crop 85%)", op_zoom(0.85)),
        ("zoom out (face 85%)", op_zoom(1 / 0.85)), ("shift 4 px", op_shift(4))], Xw, yw)

    md.extend(["## Part 3 - what the models think of YOUR microsleep images", "",
               "| Model | mean P(alert) | mean P(microsleep) | mean P(yawning) |", "|---|---|---|---|"])
    mm = yw == micro
    for arch in ("ANN", "CNN"):
        p = im.predict_proba(models[arch], Xw[mm], device).mean(0)
        md.append(f"| {arch} | " + " | ".join(f"{p[classes.index(c)]:.3f}" for c in ("alert", "microsleep", "yawning")) + " |")
        print(f"[P3] {arch} mean probs on webcam microsleep images:", {c: round(float(v), 3) for c, v in zip(classes, p)})
    md.append("")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lighting_report.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "lighting_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "condition", "model", "accuracy", "microsleep_recall"])
        w.writerows(rows)
    print(f"\nReport: {out / 'lighting_report.md'}")


if __name__ == "__main__":
    main()
