"""
Train the driver-state classifier from the CSV files in data/raw/.

    python src/train.py

* Replays every recording through WindowFeaturizer (same code as live) -> one row per 1 s stride.
* A window gets a label only if >= 80 % of its frames carry the same label ('none' is dropped).
* Evaluation is SUBJECT-INDEPENDENT when there are 2+ participants (spec section 13):
  the person being tested is never in the training part. With 1 participant the numbers are
  optimistic and the script says so.
* Saves models/driver_state_model.joblib and reports/train_report.md
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windowing import FEATURE_NAMES, WindowFeaturizer  # noqa: E402

CLASSES = ["attentive", "distracted", "drowsy"]
ROOT = Path(__file__).resolve().parents[1]


def build_dataset(csv_files, window_s=10.0, stride_s=1.0, min_purity=0.8):
    X, y, groups, blocks = [], [], [], []
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
            f = feat.compute()
            lab, purity = feat.window_label()
            if f is None or lab not in CLASSES or purity < min_purity:
                continue
            X.append([f[n] for n in FEATURE_NAMES])
            y.append(lab)
            groups.append(participant)
            blocks.append(f"{path.stem}#{int(rec['t_s'] // 30)}")
    return np.array(X), np.array(y), np.array(groups), np.array(blocks)


def make_model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=0, n_jobs=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--window-s", type=float, default=10.0)
    p.add_argument("--out", default=str(ROOT / "models" / "driver_state_model.joblib"))
    args = p.parse_args()

    files = list(Path(args.data).glob("*.csv"))
    if not files:
        sys.exit(f"No CSV files in {args.data}. Record data first: python src/collect.py --participant P01")
    X, y, groups, blocks = build_dataset(files, window_s=args.window_s)
    if len(X) == 0:
        sys.exit("No usable windows. Each recording needs at least ~20 s with a visible face and one label.")

    counts = pd.Series(y).value_counts().to_dict()
    missing = [c for c in CLASSES if c not in counts]
    if missing:
        sys.exit(f"Missing classes in the data: {missing}. Record all three labels (1, 2, 3).")
    participants = sorted(str(g) for g in set(groups))
    lines = ["# SmartCap training report", "",
             f"- recordings: {len(files)}   participants: {participants}",
             f"- windows ({args.window_s:.0f} s, 1 s stride): {len(X)}   per class: {counts}", ""]

    if len(participants) >= 3:
        splitter, split_groups = GroupKFold(n_splits=min(5, len(participants))), groups
        how = "subject-independent GroupKFold"
    elif len(participants) == 2:
        splitter, split_groups = LeaveOneGroupOut(), groups
        how = "subject-independent leave-one-participant-out"
    else:
        splitter, split_groups = GroupKFold(n_splits=min(5, len(set(blocks)))), blocks
        how = "30 s blocks of ONE participant (NOT subject-independent: numbers are optimistic)"
    lines.append(f"## Evaluation: {how}")
    print(f"Evaluation: {how}")

    y_true, y_pred = [], []
    for fold, (tr, te) in enumerate(splitter.split(X, y, split_groups), 1):
        if len(set(y[tr])) < len(CLASSES):
            continue
        m = make_model().fit(X[tr], y[tr])
        pred = m.predict(X[te])
        y_true += list(y[te])
        y_pred += list(pred)
        lines.append(f"- fold {fold}: accuracy {accuracy_score(y[te], pred):.3f} "
                     f"(test participants/blocks: {sorted(set(split_groups[te]))[:4]})")
    if y_true:
        acc, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        lines += ["", f"**accuracy {acc:.3f}   macro-F1 {f1:.3f}**", "", "```",
                  classification_report(y_true, y_pred, labels=CLASSES, zero_division=0), "```", "",
                  "Confusion matrix (rows = true, cols = predicted; order: " + ", ".join(CLASSES) + ")",
                  "```", str(cm), "```"]
        print(f"accuracy {acc:.3f}  macro-F1 {f1:.3f}")
    else:
        lines.append("Not enough data for cross-validation (every fold was missing a class).")

    final = make_model().fit(X, y)
    imp = sorted(zip(FEATURE_NAMES, final.feature_importances_), key=lambda kv: -kv[1])
    lines += ["", "## Most useful features", ""] + [f"- {n}: {v:.3f}" for n, v in imp[:10]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    params = WindowFeaturizer(window_s=args.window_s).params()
    joblib.dump({"model": final, "feature_names": FEATURE_NAMES, "classes": list(final.classes_),
                 "params": params, "trained_on": {"participants": participants, "windows": len(X)}}, out)
    report = ROOT / "reports" / "train_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Model saved: {out}\nReport saved: {report}")


if __name__ == "__main__":
    main()
