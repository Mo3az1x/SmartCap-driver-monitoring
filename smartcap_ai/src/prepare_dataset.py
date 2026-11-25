"""
Turn a downloaded image dataset (e.g. Kaggle FL3D) into the folder layout the trainer expects:

    dataset/alert/*.jpg   dataset/microsleep/*.jpg   dataset/yawning/*.jpg

STEP 1 - look at what you downloaded (paste the output to Claude if the next step fails):
    python src/prepare_dataset.py --src data/fl3d_raw --inspect

STEP 2 - organise it. The label is found, in this order, from:
    (a) a CSV file   : --csv labels.csv --path-col image --label-col label
    (b) folder names : any parent folder whose name contains a class word (alert / microsleep / yawn ...)
    (c) file names   : file name contains a class word
    (d) JSON files  : FL3D style {"<video>/frame1.jpg": {"driver_state": "alert"}}  -> used automatically when present
    python src/prepare_dataset.py --src data/fl3d_raw --dst dataset
    python src/prepare_dataset.py --src data/fl3d_raw --dst dataset --classes alert=alert,microsleep=microsleep,yawning=yawn
    (--classes  FINAL_NAME=word_to_look_for[,...] ; matching is case-insensitive "contains")

Files are COPIED (the original download is untouched). Use --max-per-class N for a quick small test set.
"""
import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = "alert=alert,microsleep=microsleep,yawning=yawn"


def parse_classes(text):
    out = {}
    for part in text.split(","):
        name, _, word = part.partition("=")
        out[name.strip()] = (word or name).strip().lower()
    return out


def inspect(src):
    files = [f for f in src.rglob("*") if f.is_file()]
    print(f"{len(files)} files under {src}")
    ext = Counter(f.suffix.lower() for f in files)
    print("File types:", dict(ext.most_common(8)))
    by_dir = Counter(str(f.parent.relative_to(src)) for f in files if f.suffix.lower() in EXT)
    print("\nFolders containing images (top 25):")
    for d, n in by_dir.most_common(25):
        print(f"  {n:>8}  {d or '.'}")
    imgs = [f for f in files if f.suffix.lower() in EXT]
    print("\nSample image names:")
    for f in imgs[:8]:
        print("  ", f.relative_to(src))
    for f in [f for f in files if f.suffix.lower() in (".csv", ".json", ".txt", ".xml")][:5]:
        print(f"\n--- {f.relative_to(src)} (first lines) ---")
        try:
            for i, line in enumerate(open(f, encoding="utf-8", errors="replace")):
                print("  ", line.rstrip()[:160])
                if i >= 4:
                    break
        except OSError:
            pass
    vids = [f for f in files if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    if vids and not imgs:
        print("\nThis download has VIDEOS but no images - tell Claude, frames must be extracted first.")


def load_json_labels(src, key="driver_state"):
    """FL3D style: {"./classification_frames/<video>/frame12.jpg": {"driver_state": "alert", ...}, ...}
    The images sit in <video>/ folders, so a key is matched by its last two parts (video folder + file name)."""
    index = {}
    for f in src.rglob("*"):
        if f.suffix.lower() in EXT:
            index[(f.parent.name, f.name)] = f
    picked, used, conflicts = {}, [], 0
    for jf in sorted(src.rglob("*.json")):
        try:
            d = json.load(open(jf, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        n = 0
        for k, v in d.items():
            if not (isinstance(v, dict) and key in v):
                continue
            parts = str(k).replace("\\", "/").split("/")
            f = index.get((parts[-2] if len(parts) > 1 else "", parts[-1]))
            if f is None:
                continue
            lab = re.sub(r"[^a-z0-9_]+", "_", str(v[key]).strip().lower())
            if f in picked:
                conflicts += picked[f] != lab
            else:
                picked[f] = lab
                n += 1
        if n:
            used.append((jf.name, n))
    return picked, used, conflicts


def safe_copy(f, target, tries=6):
    """Copy with retries: on Windows, OneDrive / antivirus can lock a file for a moment (WinError 32).
    Returns 'copied', 'exists' (already done in an earlier run) or 'failed'."""
    if target.exists() and target.stat().st_size == f.stat().st_size:
        return "exists"
    for k in range(tries):
        try:
            shutil.copyfile(f, target)
            return "copied"
        except PermissionError:
            time.sleep(0.3 * (k + 1))
        except OSError:
            break
    return "failed"


def label_from_text(text, classes):
    t = text.lower()
    hits = [name for name, word in classes.items() if word in t]
    return hits[0] if len(hits) == 1 else None


def main():
    p = argparse.ArgumentParser(description="Organise an image dataset into class folders")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", default="dataset")
    p.add_argument("--inspect", action="store_true", help="only print what is inside --src")
    p.add_argument("--classes", default=DEFAULT_CLASSES)
    p.add_argument("--csv", default=None)
    p.add_argument("--path-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--max-per-class", type=int, default=None)
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        sys.exit(f"Folder not found: {src}")
    if args.inspect:
        inspect(src)
        return

    classes = parse_classes(args.classes)
    imgs = [f for f in src.rglob("*") if f.suffix.lower() in EXT]
    if not imgs:
        sys.exit("No images found. Run with --inspect and check the folder.")

    picked = []                                    # (file, class)
    how = "folder names / file names"
    json_labels, json_used, json_conflicts = ({}, [], 0) if args.csv else load_json_labels(src)
    if json_labels:
        how = f"JSON annotations ({len(json_used)} files, key 'driver_state')"
        picked = [(f, lab) for f, lab in json_labels.items()]
        if json_conflicts:
            print(f"Note: {json_conflicts} images had different labels in different JSON files (first one kept).")
        by_video = True
    elif args.csv:
        how = f"CSV {args.csv}"
        index = {}
        for f in imgs:
            index[f.name.lower()] = f
            index[str(f.relative_to(src)).replace("\\", "/").lower()] = f
        with open(args.csv, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                key = str(row[args.path_col]).replace("\\", "/").lower()
                f = index.get(key) or index.get(Path(key).name)
                lab = label_from_text(str(row[args.label_col]), classes)
                if f is not None and lab:
                    picked.append((f, lab))
    else:
        for f in imgs:
            rel = f.relative_to(src)
            lab = label_from_text("/".join(rel.parts[:-1]), classes) or label_from_text(f.stem, classes)
            if lab:
                picked.append((f, lab))

    count = Counter(c for _, c in picked)
    by_video = bool(json_labels)
    print(f"{len(imgs)} images found, {len(picked)} matched a class using {how}: {dict(count)}")
    if not picked:
        sys.exit("Nothing matched. Run --inspect, then pass --classes NAME=word,... or --csv ... (see top of this file).")
    skipped = len(imgs) - len(picked)
    if skipped:
        print(f"{skipped} images were skipped (no class word or more than one matched).")

    groups = defaultdict(list)
    for f, c in picked:
        groups[c].append(f)
    n, done, failed = 0, 0, []
    total = sum(min(len(v), args.max_per_class or len(v)) for v in groups.values())
    print(f"Copying {total} images (safe to re-run: finished files are skipped) ...")
    for c, files in groups.items():
        files = sorted(files)
        if args.max_per_class and len(files) > args.max_per_class:
            step = len(files) / args.max_per_class                  # evenly spaced, not just the first N
            files = [files[int(i * step)] for i in range(args.max_per_class)]
        out = dst / c
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            if by_video:       # "<video>__<frame>.jpg" : the trainer splits train/val by the part before "__"
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f.parent.name) + "__" + re.sub(r"[^A-Za-z0-9_.-]+", "_", f.name)
            else:
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(f.relative_to(src).parts))
            r = safe_copy(f, out / safe)
            if r == "failed":
                failed.append(str(f))
            else:
                n += 1
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{total} processed ...")
    if failed:
        print(f"WARNING: {len(failed)} files could not be copied (locked by another program). Run the same command again; "
              f"if it repeats, pause OneDrive sync / move the project out of Desktop. First: {failed[0]}")
    print(f"Copied {n} images to {dst}  ->  {', '.join(f'{c}: {len(v)}' for c, v in groups.items())}")
    print(f"Next:  python src/train_img.py --data {dst}")


if __name__ == "__main__":
    main()
