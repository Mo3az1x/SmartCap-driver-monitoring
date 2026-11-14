"""
Image classification helpers: load a folder dataset, build an ANN (MLP) or a CNN, predict.

Dataset layout (one sub-folder per class, any names):
    dataset/
        alert/      a1.jpg a2.jpg ...
        drowsy/     d1.jpg ...
"""
import time
from pathlib import Path

import numpy as np

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- data
def read_image(path, img, gray):
    """-> uint8 array (C, H, W). np.fromfile so Windows paths with non-ASCII letters work."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.resize(im, (img, img), interpolation=cv2.INTER_AREA)
    if gray:
        return im[None, :, :]
    return np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))      # BGR->RGB, CHW


def list_images(folder, classes=None):
    folder = Path(folder)
    if classes is None:
        classes = sorted(d.name for d in folder.iterdir() if d.is_dir())
    items = []
    for ci, c in enumerate(classes):
        d = folder / c
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in EXT:
                items.append((f, ci))
    return items, list(classes)


def load_folder(folder, img=64, gray=False, classes=None, max_per_class=None, seed=0, verbose=True):
    items, classes = list_images(folder, classes)
    if not items:
        raise SystemExit(f"No images found in {folder}. Expected sub-folders per class, e.g. {folder}/drowsy/*.jpg")
    if max_per_class:
        rng = np.random.default_rng(seed)
        keep = []
        for ci in range(len(classes)):
            idx = [i for i, (_, c) in enumerate(items) if c == ci]
            rng.shuffle(idx)
            keep += idx[:max_per_class]
        items = [items[i] for i in sorted(keep)]
    X, y, paths = [], [], []
    for n, (p, c) in enumerate(items):
        a = read_image(p, img, gray)
        if a is None:
            continue
        X.append(a)
        y.append(c)
        paths.append(str(p))
        if verbose and (n + 1) % 5000 == 0:
            print(f"  loaded {n + 1}/{len(items)} images")
    return np.stack(X), np.array(y, dtype=np.int64), classes, paths


# ---------------------------------------------------------------- models
def _per_image_norm():
    """Every image is standardised on its own (mean 0, std 1): a dark / bluish room and a bright one
    look alike to the network. Lives INSIDE the model, so live code needs nothing extra."""
    import torch.nn as nn

    class PerImageNorm(nn.Module):
        def forward(self, x):
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            sd = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(0.03)
            return (x - m) / sd
    return PerImageNorm()


def build_model(arch, n_classes, img=64, channels=3, norm=True):
    import torch.nn as nn
    net = _build_body(arch, n_classes, img, channels)
    return nn.Sequential(_per_image_norm(), net) if norm else net


def _build_body(arch, n_classes, img, channels):
    import torch.nn as nn

    if arch == "ann":
        # plain fully-connected network: every pixel is an input, no notion of "neighbouring pixels"
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * img * img, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_classes))

    if arch == "cnn":
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        return nn.Sequential(
            block(channels, 32), block(32, 64), block(64, 128), block(128, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, n_classes))

    raise ValueError("arch must be 'cnn' or 'ann'")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- tensors / augmentation
def to_tensor(xb_uint8, device):
    import torch
    return torch.from_numpy(xb_uint8).to(device).float().div_(255.0)


def augment(x):
    """Batch augmentation, training only: flip, zoom / rotate / shift, gamma + brightness + contrast,
    colour cast, light noise. Makes the model less picky about camera, framing and lighting."""
    import math
    import torch
    import torch.nn.functional as F
    n, dev = x.shape[0], x.device
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    s = 0.8 + torch.rand(n, device=dev) * 0.4                         # <1 zooms in, >1 zooms out
    ang = (torch.rand(n, device=dev) - 0.5) * 2 * math.radians(8)
    tx = (torch.rand(n, device=dev) - 0.5) * 0.2
    ty = (torch.rand(n, device=dev) - 0.5) * 0.2
    c, sn = torch.cos(ang) * s, torch.sin(ang) * s
    theta = torch.stack([torch.stack([c, -sn, tx], 1), torch.stack([sn, c, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False),
                      padding_mode="border", align_corners=False)
    g = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 1.2)   # gamma ~0.55 .. 1.8
    x = x.clamp(1e-4, 1) ** g
    x = x * (0.45 + torch.rand(n, 1, 1, 1, device=dev) * 0.95)       # brightness (also darker)
    if x.shape[1] == 3:
        x = x * (0.85 + torch.rand(n, 3, 1, 1, device=dev) * 0.3)     # colour cast
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mu) * (0.7 + torch.rand(n, 1, 1, 1, device=dev) * 0.6) + mu)
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)


# ---------------------------------------------------------------- checkpoint / predict
def save_checkpoint(path, model, arch, img, gray, classes, extra=None):
    import torch
    torch.save({"arch": arch, "img": img, "gray": gray, "classes": classes,
                "state_dict": model.state_dict(), **(extra or {})}, str(path))


def load_checkpoint(path, device=None):
    import torch
    device = device or get_device()
    ck = torch.load(str(path), map_location=device, weights_only=False)
    model = build_model(ck["arch"], len(ck["classes"]), ck["img"], 1 if ck["gray"] else 3,
                        norm=ck.get("norm", False)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def predict_proba(model, X_uint8, device, batch=256):
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_uint8), batch):
            out.append(torch.softmax(model(to_tensor(X_uint8[i:i + batch], device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def time_inference(model, X_uint8, device, n=200):
    """Milliseconds per image, batch size 1 (like a live camera)."""
    import torch
    model.eval()
    X = X_uint8[:n]
    with torch.no_grad():
        for i in range(min(10, len(X))):                  # warm-up
            model(to_tensor(X[i:i + 1], device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(len(X)):
            model(to_tensor(X[i:i + 1], device))
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / max(len(X), 1) * 1000.0
