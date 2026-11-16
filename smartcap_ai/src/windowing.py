"""
Turns a stream of per-frame rows into ONE feature vector per time window (default 10 s).

The same class is used for training (replaying recorded CSVs) and for live inference,
so the model sees exactly the same kind of numbers in both cases.

Per-person differences are handled by normalising EAR / head pitch against a rolling
60 s baseline of the same person (90th percentile of EAR, median of pitch).
"""
import math
from collections import Counter, deque

import numpy as np

FEATURE_NAMES = [
    "ear_norm_mean", "ear_norm_std", "ear_norm_min", "ear_norm_p10",
    "perclos", "longest_closure_s", "long_closure_count", "blink_rate_per_min",
    "mar_mean", "mar_max", "mouth_open_frac", "longest_mouth_open_s",
    "yaw_abs_mean", "yaw_std", "turned_frac", "longest_turn_s",
    "pitch_rel_mean", "pitch_rel_std", "pitch_rel_min", "pitch_rel_max",
    "roll_abs_mean",
]

# channels of the raw sequence fed to the deep model (see WindowFeaturizer.sequence)
SEQ_CHANNELS = ["ear_norm", "mar", "yaw", "pitch_rel", "roll_div30"]


def _runs(mask, dt):
    """Durations (seconds) of consecutive True stretches."""
    out, cur = [], 0.0
    for flag, d in zip(mask, dt):
        if flag:
            cur += d
        elif cur > 0:
            out.append(cur)
            cur = 0.0
    if cur > 0:
        out.append(cur)
    return out


class WindowFeaturizer:
    def __init__(self, window_s=10.0, baseline_s=60.0, min_face_ratio=0.6, closed_ratio=0.75,
                 mar_open=0.5, turn_yaw=0.35, max_gap_s=2.0):
        self.window_s, self.baseline_s = window_s, baseline_s
        self.min_face_ratio, self.closed_ratio = min_face_ratio, closed_ratio
        self.mar_open, self.turn_yaw, self.max_gap_s = mar_open, turn_yaw, max_gap_s
        self.reset()

    def params(self):
        return dict(window_s=self.window_s, baseline_s=self.baseline_s,
                    min_face_ratio=self.min_face_ratio, closed_ratio=self.closed_ratio,
                    mar_open=self.mar_open, turn_yaw=self.turn_yaw, max_gap_s=self.max_gap_s)

    def reset(self):
        self.win = deque()        # (t, face, ear, mar, yaw, pitch, roll, label)
        self.hist = deque()       # (t, ear, pitch)  face frames only, for the baseline
        self.last_t = None
        self.no_face_since = None

    # ---------------- input ----------------
    def push(self, row, label=None):
        t = float(row["t_s"])
        if self.last_t is not None and t - self.last_t > self.max_gap_s:
            self.reset()
        self.last_t = t
        face = bool(int(row["face"]))
        g = lambda k: float(row[k]) if face else float("nan")
        self.win.append((t, face, g("ear"), g("mar"), g("yaw"), g("pitch"), g("roll"), label))
        if face:
            self.hist.append((t, g("ear"), g("pitch")))
            self.no_face_since = None
        elif self.no_face_since is None:
            self.no_face_since = t
        while self.win and self.win[0][0] < t - self.window_s:
            self.win.popleft()
        while self.hist and self.hist[0][0] < t - self.baseline_s:
            self.hist.popleft()

    # ---------------- state ----------------
    def no_face_duration(self):
        if self.no_face_since is None or self.last_t is None:
            return 0.0
        return self.last_t - self.no_face_since

    def span(self):
        return (self.win[-1][0] - self.win[0][0]) if len(self.win) > 1 else 0.0

    def face_ratio(self):
        return (sum(1 for r in self.win if r[1]) / len(self.win)) if self.win else 0.0

    def status(self):
        """'WARMUP' | 'NO_FACE' | 'OK'. NO_FACE is never turned into 'attentive'."""
        if self.span() < 0.9 * self.window_s:
            return "WARMUP"
        if self.face_ratio() < self.min_face_ratio:
            return "NO_FACE"
        return "OK"

    def window_label(self):
        labels = [r[7] for r in self.win if r[7] is not None]
        if not labels:
            return None, 0.0
        label, n = Counter(labels).most_common(1)[0]
        return label, n / len(self.win)

    # ---------------- raw sequence for the deep model ----------------
    def sequence(self, steps=100):
        """(steps, 5) float32 array: the last `window_s` seconds resampled to a fixed grid.

        Channels = SEQ_CHANNELS. EAR and pitch use the same rolling per-person baseline as compute().
        Frames without a face are bridged by linear interpolation (status() already guarantees
        that most of the window has a face)."""
        if self.status() != "OK":
            return None
        rows = [r for r in self.win if r[1]]
        if len(rows) < 2:
            return None
        base_ear = float(np.percentile([h[1] for h in self.hist], 90))
        base_pitch = float(np.median([h[2] for h in self.hist]))
        if base_ear <= 1e-6:
            return None
        t = np.array([r[0] for r in rows])
        end = self.win[-1][0]
        grid = np.linspace(end - self.window_s, end, steps)

        def channel(i, fn=lambda v: v):
            return np.interp(grid, t, fn(np.array([r[i] for r in rows])))

        seq = np.stack([channel(2, lambda v: v / base_ear), channel(3), channel(4),
                        channel(5, lambda v: v - base_pitch), channel(6, lambda v: v / 30.0)],
                       axis=1).astype(np.float32)
        return seq if np.isfinite(seq).all() else None

    # ---------------- features ----------------
    def compute(self):
        if self.status() != "OK":
            return None
        rows = [r for r in self.win if r[1]]
        t = np.array([r[0] for r in rows])
        ear, mar = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
        yaw, pitch, roll = (np.array([r[4] for r in rows]), np.array([r[5] for r in rows]),
                            np.array([r[6] for r in rows]))
        base_ear = float(np.percentile([h[1] for h in self.hist], 90))
        base_pitch = float(np.median([h[2] for h in self.hist]))
        if base_ear <= 1e-6:
            return None

        dt = np.diff(t, prepend=t[0])
        dt = np.clip(dt, 0.0, 0.5)
        pos = dt[dt > 0]
        dt[0] = float(np.median(pos)) if len(pos) else 0.03
        total = float(dt.sum())
        if total <= 0:
            return None

        ear_n = ear / base_ear
        closed = ear_n < self.closed_ratio
        closures = _runs(closed, dt)
        blinks = [c for c in closures if 0.05 <= c <= 0.5]
        mouth = _runs(mar > self.mar_open, dt)
        turned = np.abs(yaw) > self.turn_yaw
        turns = _runs(turned, dt)
        pitch_rel = pitch - base_pitch

        f = {
            "ear_norm_mean": ear_n.mean(), "ear_norm_std": ear_n.std(),
            "ear_norm_min": ear_n.min(), "ear_norm_p10": np.percentile(ear_n, 10),
            "perclos": float((dt * closed).sum() / total),
            "longest_closure_s": max(closures, default=0.0),
            "long_closure_count": sum(1 for c in closures if c > 0.5),
            "blink_rate_per_min": len(blinks) * 60.0 / total,
            "mar_mean": mar.mean(), "mar_max": mar.max(),
            "mouth_open_frac": float((dt * (mar > self.mar_open)).sum() / total),
            "longest_mouth_open_s": max(mouth, default=0.0),
            "yaw_abs_mean": np.abs(yaw).mean(), "yaw_std": yaw.std(),
            "turned_frac": float((dt * turned).sum() / total),
            "longest_turn_s": max(turns, default=0.0),
            "pitch_rel_mean": pitch_rel.mean(), "pitch_rel_std": pitch_rel.std(),
            "pitch_rel_min": pitch_rel.min(), "pitch_rel_max": pitch_rel.max(),
            "roll_abs_mean": np.abs(roll).mean(),
        }
        f = {k: float(v) for k, v in f.items()}
        if any(math.isnan(v) for v in f.values()):
            return None
        return f
