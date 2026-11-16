"""
Train an image classifier twice on the SAME 80% / 20% split: a plain ANN (MLP) and a CNN.
Prints loss / accuracy / time per epoch, saves curves, confusion matrices, a comparison report.

    python src/train_img.py --data dataset                  # both ANN and CNN
    python src/train_img.py --data dataset --arch cnn --epochs 30 --img 64
    python src/train_img.py --data dataset --gray            # 1-channel images (faster, 3x less memory)

Dataset: one sub-folder per class (dataset/alert/*.jpg, dataset/drowsy/*.jpg, ...).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgmodels as im  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def split_indices(paths, y, test_size=0.2, seed=0):
    """80/20 split. If file names look like '<video>__<frame>.jpg' the split is BY VIDEO (all frames of a
    video go to the same side), so near-identical neighbouring frames cannot leak into validation."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    groups = np.array([Path(p).name.split("__")[0] if "__" in Path(p).name else Path(p).name for p in paths])
    if len(set(groups)) == len(groups):
        tr, va = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
        return tr, va, False
    total = np.bincount(y, minlength=y.max() + 1)
    best, best_score = None, 1e9
    for s in range(seed, seed + 300):               # pick the split whose class mix is closest to 80/20
        tr, va = next(GroupShuffleSplit(1, test_size=test_size, random_state=s).split(np.zeros(len(y)), y, groups))
        frac = np.bincount(y[va], minlength=len(total)) / np.maximum(total, 1)
        score = np.abs(frac - test_size).max()
        if score < best_score:
            best, best_score = (tr, va), score
    return best[0], best[1], True


def run_experiment(arch, X, y, classes, tr_idx, va_idx, img, gray, epochs=20, batch=64, lr=1e-3,
                   seed=0, out_dir=None, model_dir=None, verbose=True):
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = im.get_device()
    ch = 1 if gray else 3
    model = im.build_model(arch, len(classes), img, ch).to(device)
    n_params = im.count_params(model)

    counts = np.bincount(y[tr_idx], minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr_idx) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=steps)

    Xtr, ytr, Xva, yva = X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]
    hist = []
    best_acc, best_state = -1.0, None
    t_all = time.perf_counter()
    if verbose:
        print(f"\n=== {arch.upper()} | params {n_params:,} | device {device} | train {len(tr_idx)} / val {len(va_idx)} ===")
        print(f"{'epoch':>5} {'train_loss':>10} {'train_acc':>9} {'val_loss':>9} {'val_acc':>8} {'time_s':>7}")

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = np.random.permutation(len(Xtr))
        tl, tc = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            xb = im.augment(im.to_tensor(Xtr[b], device))
            yb = torch.from_numpy(ytr[b]).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
            tc += (out.argmax(1) == yb).sum().item()
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(Xva), 256):
                xb = im.to_tensor(Xva[i:i + 256], device)
                yb = torch.from_numpy(yva[i:i + 256]).to(device)
                out = model(xb)
                vl += loss_fn(out, yb).item() * len(yb)
                vc += (out.argmax(1) == yb).sum().item()
        row = {"epoch": ep, "train_loss": tl / len(Xtr), "train_acc": tc / len(Xtr),
               "val_loss": vl / len(Xva), "val_acc": vc / len(Xva), "time_s": time.perf_counter() - t0}
        hist.append(row)
        if row["val_acc"] > best_acc:
            best_acc, best_state = row["val_acc"], copy.deepcopy(model.state_dict())
        if verbose:
            print(f"{ep:>5} {row['train_loss']:>10.4f} {row['train_acc']:>9.4f} {row['val_loss']:>9.4f} "
                  f"{row['val_acc']:>8.4f} {row['time_s']:>7.1f}")

    train_time = time.perf_counter() - t_all
    model.load_state_dict(best_state)
    probs = im.predict_proba(model, Xva, device)
    pred = probs.argmax(1)
    cm = confusion_matrix(yva, pred, labels=range(len(classes)))
    rep = classification_report(yva, pred, labels=range(len(classes)), target_names=classes,
                                zero_division=0, output_dict=True)
    ms_img = im.time_inference(model, Xva, device)
    result = {"arch": arch, "params": n_params, "device": str(device), "train_time_s": train_time,
              "best_val_acc": best_acc, "final_val_acc": hist[-1]["val_acc"], "final_val_loss": hist[-1]["val_loss"],
              "final_train_acc": hist[-1]["train_acc"], "final_train_loss": hist[-1]["train_loss"],
              "ms_per_image": ms_img, "macro_f1": rep["macro avg"]["f1-score"],
              "confusion": cm.tolist(), "history": hist}
    if verbose:
        print(f"{arch.upper()}: best val acc {best_acc:.4f} | train time {train_time:.0f} s | "
              f"inference {ms_img:.2f} ms/image (batch 1, {device})")

    if model_dir:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        im.save_checkpoint(Path(model_dir) / f"img_{arch}.pt", model, arch, img, gray, classes,
                           {"best_val_acc": best_acc, "norm": True})
    if out_dir:
        save_outputs(result, classes, Path(out_dir))
    return result, model


def save_outputs(result, classes, out_dir):
    import csv
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    arch, hist = result["arch"], result["history"]
    with open(out_dir / f"{arch}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="validation")
    ax[0].set(title=f"{arch.upper()} loss", xlabel="epoch", ylabel="loss")
    ax[1].plot(ep, [h["train_acc"] * 100 for h in hist], label="train")
    ax[1].plot(ep, [h["val_acc"] * 100 for h in hist], label="validation")
    ax[1].set(title=f"{arch.upper()} accuracy (%)", xlabel="epoch", ylabel="%")
    ax[2].bar(ep, [h["time_s"] for h in hist])
    ax[2].set(title=f"{arch.upper()} time per epoch (s)", xlabel="epoch", ylabel="seconds")
    for a in ax[:2]:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{arch}_curves.png", dpi=130)
    plt.close(fig)

    cm = np.array(result["confusion"])
    fig, a = plt.subplots(figsize=(1.2 * len(classes) + 3, 1.2 * len(classes) + 2.5))
    a.imshow(cm, cmap="Blues")
    a.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
          xlabel="predicted", ylabel="true", title=f"{arch.upper()} confusion matrix (validation)")
    plt.setp(a.get_xticklabels(), rotation=30, ha="right")
    for i in range(len(classes)):
        for j in range(len(classes)):
            a.text(j, i, cm[i, j], ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / f"{arch}_confusion.png", dpi=130)
    plt.close(fig)


def write_report(results, classes, n_train, n_val, args, path, by_video=False):
    lines = ["# Image models: ANN vs CNN", "",
             f"- Data: {n_train + n_val} images, classes: {', '.join(classes)}",
             f"- Split: {n_train} train (80%) / {n_val} validation (20%), same split for both models, "
             + ("done BY VIDEO (all frames of a video are on one side only)" if by_video else "random per image"),
             f"- Image size: {args.img}x{args.img}, {'grayscale' if args.gray else 'RGB'}; epochs: {args.epochs}; batch: {args.batch}",
             "", "| Model | Params | Val acc | Val loss | Train acc | Train loss | Macro F1 | Train time | ms / image |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['arch'].upper()} | {r['params']:,} | {r['best_val_acc']*100:.2f}% | {r['final_val_loss']:.4f} | "
                     f"{r['final_train_acc']*100:.2f}% | {r['final_train_loss']:.4f} | {r['macro_f1']:.3f} | "
                     f"{r['train_time_s']:.0f} s | {r['ms_per_image']:.2f} |")
    lines += ["", "Val acc = best epoch on the validation set; loss/train columns = last epoch."]
    if by_video:
        lines.append("Note: video ids are not person ids - the same driver may appear in several videos, so the numbers can "
                     "still be a little optimistic. Testing on a person / camera the model never saw is the real check.")
    else:
        lines.append("**Honest note:** the split is random per image. If frames of the same person appear in both train and "
                     "validation, the numbers are optimistic.")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def compare_plot(results, out_png):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in results:
        ep = [h["epoch"] for h in r["history"]]
        ax[0].plot(ep, [h["val_loss"] for h in r["history"]], label=r["arch"].upper())
        ax[1].plot(ep, [h["val_acc"] * 100 for h in r["history"]], label=r["arch"].upper())
    ax[0].set(title="validation loss", xlabel="epoch")
    ax[1].set(title="validation accuracy (%)", xlabel="epoch")
    ax[2].bar([r["arch"].upper() for r in results], [r["train_time_s"] for r in results])
    ax[2].set(title="total training time (s)")
    for a in ax[:2]:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")

    p = argparse.ArgumentParser(description="Train ANN and CNN on an image folder (80/20 split)")
    p.add_argument("--data", required=True, help="folder with one sub-folder per class")
    p.add_argument("--arch", default="both", choices=["both", "cnn", "ann"])
    p.add_argument("--img", type=int, default=64, help="images are resized to img x img")
    p.add_argument("--gray", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-per-class", type=int, default=None, help="limit images per class (quick tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "reports" / "img"))
    p.add_argument("--models", default=str(ROOT / "models"))
    args = p.parse_args()

    print("Loading images ...")
    t0 = time.perf_counter()
    X, y, classes, X_paths = im.load_folder(args.data, args.img, args.gray, max_per_class=args.max_per_class, seed=args.seed)
    print(f"{len(X)} images, classes {classes}, counts {np.bincount(y).tolist()}, loaded in {time.perf_counter()-t0:.0f} s")
    tr, va, by_video = split_indices(X_paths, y, 0.2, args.seed)
    print(f"split {'BY VIDEO' if by_video else 'random'}: train {len(tr)} / val {len(va)} | "
          f"val per class {np.bincount(y[va], minlength=len(classes)).tolist()}")

    archs = ["ann", "cnn"] if args.arch == "both" else [args.arch]
    results = []
    for a in archs:
        r, _ = run_experiment(a, X, y, classes, tr, va, args.img, args.gray, args.epochs, args.batch, args.lr,
                              args.seed, args.out, args.models)
        results.append(r)
    out = Path(args.out)
    write_report(results, classes, len(tr), len(va), args, out / "img_compare.md", by_video)
    compare_plot(results, out / "img_compare.png")
    (out / "img_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nReport: {out / 'img_compare.md'}\nPlots:  {out}\nModels: {Path(args.models) / 'img_<arch>.pt'}")


if __name__ == "__main__":
    main()
