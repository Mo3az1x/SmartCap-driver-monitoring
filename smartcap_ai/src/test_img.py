"""
Test trained image models (ANN and/or CNN) on YOUR OWN images.

    # your images in class sub-folders (we can compute accuracy):
    python src/test_img.py --images my_images --models models/img_cnn.pt models/img_ann.pt

    # a flat folder of unlabeled images (only predictions are written):
    python src/test_img.py --images some_folder --models models/img_cnn.pt

Writes reports/img_test/predictions_<arch>.csv (+ confusion matrix PNG when labels exist).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgmodels as im  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_any(folder, img, gray, classes):
    """Class sub-folders -> labelled; otherwise a flat folder -> y = -1."""
    folder = Path(folder)
    subs = [d.name for d in folder.iterdir() if d.is_dir()]
    known = [c for c in classes if c in subs]
    if known:
        X, y, _, paths = im.load_folder(folder, img, gray, classes=classes, verbose=False)
        return X, y, paths
    files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in im.EXT)
    X, paths = [], []
    for f in files:
        a = im.read_image(f, img, gray)
        if a is not None:
            X.append(a)
            paths.append(str(f))
    if not X:
        raise SystemExit(f"No images in {folder}")
    return np.stack(X), np.full(len(X), -1), paths


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", default=str(ROOT / "reports" / "img_test"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = im.get_device()

    for mp in args.models:
        model, ck = im.load_checkpoint(mp, device)
        classes, arch = ck["classes"], ck["arch"]
        X, y, paths = load_any(args.images, ck["img"], ck["gray"], classes)
        t0 = time.perf_counter()
        probs = im.predict_proba(model, X, device)
        dt = (time.perf_counter() - t0) / len(X) * 1000
        pred = probs.argmax(1)
        print(f"\n=== {arch.upper()} on {len(X)} images ({dt:.2f} ms/image batched) ===")
        with open(out / f"predictions_{arch}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "true", "predicted", "confidence"] + [f"p_{c}" for c in classes])
            for i, pth in enumerate(paths):
                w.writerow([pth, classes[y[i]] if y[i] >= 0 else "", classes[pred[i]], f"{probs[i].max():.3f}"]
                           + [f"{v:.3f}" for v in probs[i]])
        if (y >= 0).any():
            m = y >= 0
            acc = (pred[m] == y[m]).mean()
            print(f"Accuracy: {acc*100:.2f}%")
            print(classification_report(y[m], pred[m], labels=range(len(classes)), target_names=classes, zero_division=0))
            cm = confusion_matrix(y[m], pred[m], labels=range(len(classes)))
            fig, a = plt.subplots(figsize=(1.2 * len(classes) + 3, 1.2 * len(classes) + 2.5))
            a.imshow(cm, cmap="Blues")
            a.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
                  xlabel="predicted", ylabel="true", title=f"{arch.upper()} on your images: {acc*100:.1f}%")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            fig.tight_layout()
            fig.savefig(out / f"test_{arch}_confusion.png", dpi=130)
            plt.close(fig)
        else:
            counts = {classes[c]: int((pred == c).sum()) for c in range(len(classes))}
            print("Predicted counts:", counts)
    print(f"\nFiles in {out}")


if __name__ == "__main__":
    main()
