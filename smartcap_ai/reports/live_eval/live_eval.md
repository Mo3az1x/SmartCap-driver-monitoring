# Live evaluation (CNN)

- Labelled frames counted: 1130 (frames in the first 1.0 s after each key press, paused frames and no-face frames [360] are not counted)
- **Per-frame accuracy** (raw CNN, no smoothing): **66.1%**
- **State accuracy** (after smoothing + rules; UNKNOWN counts as wrong): **65.6%**
- UNKNOWN rate: 2.3%
- Time per frame (face crop -> CNN -> state, this machine): mean 29.2 ms, p95 37.1 ms

| true state | frames | per-frame acc | state acc | UNKNOWN |
|---|---|---|---|---|
| alert | 375 | 100.0% | 100.0% | 0.0% |
| microsleep | 374 | 72.2% | 68.4% | 5.3% |
| yawning | 381 | 26.8% | 28.9% | 1.6% |

Confusion matrix (rows: what you acted, columns: what the CNN said, per frame):

| true \ predicted | alert | microsleep | yawning |
|---|---|---|---|
| alert | 375 | 0 | 0 |
| microsleep | 104 | 270 | 0 |
| yawning | 265 | 14 | 102 |

Honest notes: you are the only subject, the labels are what you *meant* to act (a half-closed eye while labelled 'alert' counts as an error), and neighbouring frames are almost identical, so treat this as a demo check, not a final benchmark.