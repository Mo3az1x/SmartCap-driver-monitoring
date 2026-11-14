"""
Train a DEEP LEARNING model on the raw 10 s sequences and compare it with the RandomForest on
exactly the same subject-independent folds.

    python src/train_dl.py                                  # small CNN+GRU (~13k params), fine on a laptop CPU
    python src/train_dl.py --arch large --epochs 80 --seeds 5   # big model + ensemble, meant for a GPU (Colab)

Outputs (small  -> driver_state_dl.*,  large -> driver_state_dl_large.*)
    models/<name>.onnx   the model (runs with onnxruntime only - also on the Raspberry Pi)
    models/<name>.json   classes / window settings needed by src/engine.py  (keep it next to the .onnx)
    models/<name>.pt     PyTorch weights
    reports/train_dl_report.md   (large: train_dl_large_report.md)

Use it live:  python src/live.py --model models/driver_state_dl_large.onnx
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CLASSES, make_model  # noqa: E402
from windowing import FEATURE_NAMES, SEQ_CHANNELS, WindowFeaturizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEPS = 100          # 10 s at 10 Hz
N_CH = len(SEQ_CHANNELS)


class Normalised(nn.Module):
    """Base class: per-channel mean/std stored INSIDE the model (so ONNX needs raw units only)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 1, N_CH))
        self.register_buffer("std", torch.ones(1, 1, N_CH))


class DriverNet(Normalised):
    """small: ~13k parameters. Two Conv1d blocks find short patterns (blink, yawn, head turn),
    the GRU looks at how they follow each other over the 10 s window."""

    def __init__(self, in_ch=N_CH, n_classes=len(CLASSES)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(32, n_classes)

    def forward(self, x):                       # x: (batch, steps, channels), raw units
        x = (x - self.mean) / self.std
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, h = self.gru(x)
        return self.head(self.drop(h[-1]))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, p=0.15):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, 5, padding=2), nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(cout, cout, 5, padding=2), nn.BatchNorm1d(cout))
        self.skip = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class DriverNetLarge(Normalised):
    """large: ~650k parameters. ResNet-1D feature extractor -> 2-layer bidirectional GRU ->
    attention pooling over time (the network learns WHICH moments of the 10 s matter) -> MLP head."""

    def __init__(self, in_ch=N_CH, n_classes=len(CLASSES)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.res = nn.Sequential(ResBlock(64, 64), ResBlock(64, 128), nn.MaxPool1d(2),
                                 ResBlock(128, 128), nn.MaxPool1d(2))
        self.gru = nn.GRU(128, 96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.att = nn.Linear(192, 1)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.res(self.stem(x.transpose(1, 2))).transpose(1, 2)      # (B, 25, 128)
        h, _ = self.gru(x)                                              # (B, 25, 192)
        w = torch.softmax(self.att(h), dim=1)                           # attention over time
        return self.head((w * h).sum(dim=1))


class Ensemble(nn.Module):
    """Averages the probabilities of several independently trained nets (different seeds).
    Returns log-probabilities, so softmax() in the engine gives back the averaged probabilities."""

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        p = torch.stack([torch.softmax(n(x), dim=1) for n in self.nets]).mean(0)
        return torch.log(p + 1e-8)


ARCHS = {"small": DriverNet, "large": DriverNetLarge}
HYPER = {"small": dict(lr=3e-3, wd=1e-3, smooth=0.0, strong_aug=False),
         "large": dict(lr=1e-3, wd=1e-2, smooth=0.05, strong_aug=True)}


def build_dataset(csv_files, window_s=10.0, stride_s=1.0, min_purity=0.8):
    Xf, Xs, y, groups, blocks = [], [], [], [], []
    for path in sorted(csv_files):
        df = pd.read_csv(path)
        participant = str(df["participant"].iloc[0]) if "participant" in df else path.stem.split("_")[0]
        feat = WindowFeaturizer(window_s=window_s)
        next_t = None
        for rec in df.to_dict("records"):
            label = rec.get("label")
            feat.push(rec, None if label in (None, "none") else label)
            if next_t is None:
                next_t = rec["t_s"] + window_s
            if rec["t_s"] < next_t:
                continue
            next_t = rec["t_s"] + stride_s
            f, s = feat.compute(), feat.sequence(STEPS)
            lab, purity = feat.window_label()
            if f is None or s is None or lab not in CLASSES or purity < min_purity:
                continue
            Xf.append([f[n] for n in FEATURE_NAMES])
            Xs.append(s)
            y.append(CLASSES.index(lab))
            groups.append(participant)
            blocks.append(f"{path.stem}#{int(rec['t_s'] // 30)}")
    return np.array(Xf), np.array(Xs), np.array(y), np.array(groups), np.array(blocks)


def augment(x, strong=False):
    """Augmentation (training only) - the main defence against over-fitting on little data."""
    b, steps, _ = x.shape
    scale = x.new_tensor([1.0, 0.2, 0.5, 0.2, 0.2]).view(1, 1, -1)
    x = x + 0.02 * torch.randn_like(x) * scale
    x[:, :, 0] = x[:, :, 0] * (0.9 + 0.2 * torch.rand(b, 1, device=x.device))       # eye-size differences
    x[:, :, 3] = x[:, :, 3] + 0.02 * torch.randn(b, 1, device=x.device)             # pitch offset
    flip = (torch.rand(b, 1, device=x.device) < 0.5).float() * 2 - 1                # mirror left/right
    x[:, :, 2] = x[:, :, 2] * flip
    x[:, :, 4] = x[:, :, 4] * flip
    if strong:                                    # simulate short tracking drop-outs (signal frozen 0.3-1.5 s)
        for i in torch.nonzero(torch.rand(b, device=x.device) < 0.3).flatten().tolist():
            length = int(torch.randint(3, 16, (1,)))
            start = int(torch.randint(1, steps - length, (1,)))
            x[i, start:start + length, :] = x[i, start - 1:start, :]
    return x


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_net(Xs, y, epochs, arch="small", seed=0, device=None):
    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    hp = HYPER[arch]
    net = ARCHS[arch]()
    flat = Xs.reshape(-1, Xs.shape[-1])
    net.mean.copy_(torch.tensor(flat.mean(0)).view(1, 1, -1))
    net.std.copy_(torch.tensor(flat.std(0) + 1e-6).view(1, 1, -1))
    net.to(device)
    xt, yt = torch.tensor(Xs).to(device), torch.tensor(y).to(device)
    weights = torch.tensor(len(y) / (len(CLASSES) * np.maximum(np.bincount(y, minlength=len(CLASSES)), 1)),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=hp["smooth"])
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(net(augment(xt[idx].clone(), hp["strong_aug"])), yt[idx]).backward()
            opt.step()
        sched.step()
    return net.eval()


def predict(net, Xs, device=None):
    device = device or next(net.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 512):
            out.append(net(torch.tensor(Xs[i:i + 512]).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def export_onnx(net, path):
    net = net.cpu().eval()
    # batch size is fixed to 1: the live engine always scores one window at a time
    dummy = torch.zeros(1, STEPS, N_CH)
    kwargs = dict(input_names=["seq"], output_names=["logits"], opset_version=17)
    try:
        torch.onnx.export(net, (dummy,), str(path), **kwargs)
    except Exception as err:                       # older / newer torch exporters differ
        print(f"default ONNX exporter failed ({type(err).__name__}), retrying with legacy exporter")
        torch.onnx.export(net, (dummy,), str(path), dynamo=False, **kwargs)
    # newer exporters may put the weights in a side file (*.onnx.data): fold them back into ONE file
    import onnx
    side = Path(str(path) + ".data")
    if side.exists():
        onnx.save(onnx.load(str(path)), str(path), save_as_external_data=False)
        side.unlink()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probe = np.random.default_rng(0).normal(1.0, 0.3, (4, STEPS, N_CH)).astype(np.float32)
    for sample in probe:
        got = sess.run(None, {"seq": sample[None]})[0]
        with torch.no_grad():
            want = net(torch.tensor(sample[None])).numpy()
        assert np.allclose(got, want, atol=1e-3), "ONNX output differs from PyTorch!"
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(None, {"seq": probe[:1]})
    return (time.perf_counter() - t0) / 50 * 1000.0          # ms per window on this CPU


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--window-s", type=float, default=10.0)
    p.add_argument("--arch", choices=list(ARCHS), default="small")
    p.add_argument("--epochs", type=int, default=None, help="default: 40 (small) / 80 (large)")
    p.add_argument("--seeds", type=int, default=1, help="final model = ensemble of N seeds (CV uses 1)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    epochs = args.epochs or (40 if args.arch == "small" else 80)
    name = "driver_state_dl" if args.arch == "small" else f"driver_state_dl_{args.arch}"
    out = Path(args.out) if args.out else ROOT / "models" / f"{name}.onnx"
    report_name = "train_dl_report.md" if args.arch == "small" else f"train_dl_{args.arch}_report.md"
    device = get_device()

    files = list(Path(args.data).glob("*.csv"))
    if not files:
        sys.exit(f"No CSV files in {args.data}. Record data first: python src/collect.py --participant P01")
    Xf, Xs, y, groups, blocks = build_dataset(files, window_s=args.window_s)
    if len(y) == 0 or len(set(y)) < len(CLASSES):
        sys.exit("Need usable windows for all three labels (1 attentive, 2 distracted, 3 drowsy).")
    participants = sorted(str(g) for g in set(groups))
    counts = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}

    if len(participants) >= 3:
        splitter, sg, how = GroupKFold(n_splits=min(5, len(participants))), groups, "subject-independent GroupKFold"
    elif len(participants) == 2:
        splitter, sg, how = LeaveOneGroupOut(), groups, "subject-independent leave-one-participant-out"
    else:
        splitter, sg = GroupKFold(n_splits=min(5, len(set(blocks)))), blocks
        how = "30 s blocks of ONE participant (NOT subject-independent: numbers are optimistic)"
    n_params = sum(p.numel() for p in ARCHS[args.arch]().parameters())
    print(f"device: {device} | arch: {args.arch} ({n_params:,} params) | epochs: {epochs} | seeds: {args.seeds}")
    print(f"windows: {len(y)} {counts} | evaluation: {how}")

    t0 = time.time()
    truth, pred_dl, pred_rf = [], [], []
    for fold, (tr, te) in enumerate(splitter.split(Xs, y, sg), 1):
        if len(set(y[tr])) < len(CLASSES):
            continue
        net = fit_net(Xs[tr], y[tr], epochs, args.arch, device=device)
        rf = make_model().fit(Xf[tr], y[tr])
        truth += list(y[te])
        pred_dl += list(predict(net, Xs[te], device))
        pred_rf += list(rf.predict(Xf[te]))
        print(f"  fold {fold}: DL {accuracy_score(y[te], pred_dl[-len(te):]):.3f} | "
              f"RF {accuracy_score(y[te], pred_rf[-len(te):]):.3f}", flush=True)

    lines = ["# SmartCap deep-learning report", "",
             f"- participants: {participants}   windows: {len(y)}   per class: {counts}",
             f"- evaluation: {how}",
             f"- arch: {args.arch} ({n_params:,} params)   epochs: {epochs}   final ensemble seeds: {args.seeds}"
             f"   device: {device}", ""]
    if truth:
        label = f"1D-CNN/GRU {args.arch} (deep learning, 1 seed in CV)"
        rows = [f"| {name_} | {accuracy_score(truth, pred):.3f} | {f1_score(truth, pred, average='macro'):.3f} |"
                for name_, pred in ((label, pred_dl), ("RandomForest (baseline)", pred_rf))]
        lines += ["| model | accuracy | macro-F1 |", "|---|---|---|"] + rows + ["",
                  "DL confusion matrix (rows = true, cols = predicted; order: " + ", ".join(CLASSES) + ")",
                  "```", str(confusion_matrix(truth, pred_dl, labels=range(len(CLASSES)))), "```", "```",
                  classification_report(truth, pred_dl, target_names=CLASSES, zero_division=0), "```"]
        print("\n".join(["| model | accuracy | macro-F1 |"] + rows))
    else:
        lines.append("Not enough data for cross-validation.")

    nets = [fit_net(Xs, y, epochs, args.arch, seed=s, device=device) for s in range(args.seeds)]
    final = nets[0] if args.seeds == 1 else Ensemble([n.cpu() for n in nets]).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    ms = export_onnx(final, out)
    torch.save(final.state_dict(), out.with_suffix(".pt"))
    meta = {"classes": CLASSES, "steps": STEPS, "channels": SEQ_CHANNELS, "arch": args.arch,
            "seeds": args.seeds, "params": WindowFeaturizer(window_s=args.window_s).params(),
            "trained_on": {"participants": participants, "windows": int(len(y))}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines += ["", f"ONNX inference on this machine's CPU: {ms:.1f} ms per 10 s window "
                  f"(the engine scores 2 windows per second)",
              f"Total training time: {(time.time() - t0) / 60:.1f} min"]
    report = ROOT / "reports" / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out} (+ .json, .pt) | ONNX {ms:.1f} ms/window\nReport: {report}")


if __name__ == "__main__":
    main()
