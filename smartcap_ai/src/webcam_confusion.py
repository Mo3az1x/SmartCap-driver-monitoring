"""
Confusion matrices of the trained ANN and CNN on YOUR webcam images (PNG, no .tex).

Run from smartcap_ai/ (put this file in src/):
    python src/webcam_confusion.py --images my_images --models models\img_ann.pt models\img_cnn.pt --out reports\webcam_cm

Writes: cm_ann.png, cm_cnn.png, cm_both.png (+ .pdf) and prints the counts and per-class recall.
Use the checkpoints you want in the paper (FL3D-only models, not trained with --extra).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgmodels as im  # noqa: E402


def draw(ax, cm, classes, title):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(1).max())
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=12)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--models", nargs=2, required=True, help="ANN checkpoint then CNN checkpoint")
    ap.add_argument("--out", default="reports/webcam_cm")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = im.get_device()
    res = {}
    for path in a.models:
        m, ck = im.load_checkpoint(path, device)
        arch, img, gray, classes = ck["arch"].upper(), ck["img"], ck["gray"], ck["classes"]
        X, y, _, _ = im.load_folder(a.images, img, gray, classes=classes, verbose=False)
        pred = im.predict_proba(m, X, device).argmax(1)
        k = len(classes)
        cm = np.zeros((k, k), int)
        for t, p in zip(y, pred):
            cm[t, p] += 1
        acc = np.trace(cm) / cm.sum()
        res[arch] = (cm, acc)
        print(f"\n{arch}: accuracy {acc*100:.1f}% on {cm.sum()} images")
        print(cm)
        for i, c in enumerate(classes):
            print(f"  recall {c}: {cm[i, i] / max(cm[i].sum(), 1) * 100:.1f}%")
        fig, ax = plt.subplots(figsize=(4.2, 4))
        draw(ax, cm, classes, f"{arch} on webcam images: {acc*100:.1f}%")
        fig.tight_layout()
        fig.savefig(out / f"cm_{arch.lower()}.png", dpi=300)
        fig.savefig(out / f"cm_{arch.lower()}.pdf")
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 4))
    for ax, arch in zip(axs, ("ANN", "CNN")):
        cm, acc = res[arch]
        draw(ax, cm, classes, f"{arch}: {acc*100:.1f}%")
    fig.tight_layout()
    fig.savefig(out / "cm_both.png", dpi=300)
    fig.savefig(out / "cm_both.pdf")
    print("\nSaved to", out.resolve())


if __name__ == "__main__":
    main()
