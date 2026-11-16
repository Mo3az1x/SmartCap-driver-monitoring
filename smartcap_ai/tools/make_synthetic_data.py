"""
FAKE data generator - ONLY to test that the pipeline runs end to end without a camera.
The numbers are invented, so the accuracy it gives means NOTHING about real drivers.
Never mix this output with real recordings in data/raw/.

    python tools/make_synthetic_data.py --out /tmp/fake_raw --participants 4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 15


def segment(label, seconds, base_ear, base_pitch, rng, t0):
    n = int(seconds * FPS)
    t = t0 + np.arange(n) / FPS
    ear = base_ear + rng.normal(0, 0.008, n)
    mar = 0.05 + np.abs(rng.normal(0, 0.02, n))
    yaw = rng.normal(0, 0.04, n)
    pitch = base_pitch + rng.normal(0, 0.006, n)

    def event(start_s, dur_s, fn):
        a, b = int(start_s * FPS), int(min(seconds, start_s + dur_s) * FPS)
        if a < n:
            fn(slice(a, b))

    s = rng.uniform(1, 4)
    while s < seconds:                       # normal blinks (all classes)
        event(s, 0.2, lambda sl: ear.__setitem__(sl, base_ear * 0.3))
        s += rng.uniform(2.5, 6)

    if label == "distracted":
        s = rng.uniform(2, 5)
        while s < seconds:
            d = rng.uniform(1.5, 3.0)
            sign = rng.choice([-1, 1])
            event(s, d, lambda sl: yaw.__setitem__(sl, sign * rng.uniform(0.45, 0.7)))
            if rng.random() < 0.4:
                event(s, d, lambda sl: pitch.__setitem__(sl, base_pitch + 0.09))
            s += d + rng.uniform(3, 7)
    elif label == "drowsy":
        ear *= 0.9
        s = rng.uniform(2, 6)
        while s < seconds:                   # long closures
            event(s, rng.uniform(1.0, 3.0), lambda sl: ear.__setitem__(sl, base_ear * 0.25))
            s += rng.uniform(5, 11)
        s = rng.uniform(5, 15)
        while s < seconds:                   # yawns
            event(s, rng.uniform(1.4, 2.5), lambda sl: mar.__setitem__(sl, rng.uniform(0.6, 0.8)))
            s += rng.uniform(15, 30)
        s = rng.uniform(3, 9)
        while s < seconds:                   # nods
            event(s, 1.2, lambda sl: pitch.__setitem__(sl, base_pitch + 0.08))
            s += rng.uniform(6, 14)

    return pd.DataFrame({
        "t_s": t, "face": 1, "ear": ear, "ear_l": ear, "ear_r": ear, "mar": mar, "yaw": yaw,
        "pitch": pitch, "roll": rng.normal(0, 2, n), "face_w": 0.25, "brightness": 120.0,
        "label": label}), t[-1] + 1 / FPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--participants", type=int, default=4)
    p.add_argument("--seconds", type=float, default=150)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.participants + 1):
        rng = np.random.default_rng(i)
        base_ear, base_pitch = rng.uniform(0.24, 0.34), rng.uniform(0.5, 0.6)
        parts, t0 = [], 0.0
        for label in ("attentive", "distracted", "drowsy"):
            df, t0 = segment(label, args.seconds, base_ear, base_pitch, rng, t0)
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        df["participant"] = f"P{i:02d}"
        df.to_csv(out / f"P{i:02d}_synthetic.csv", index=False)
    print(f"Wrote {args.participants} fake recordings to {out}")


if __name__ == "__main__":
    main()
