# Live evaluation (CNN)

- Labelled frames counted: 855 (frames in the first 1.0 s after each key press, paused frames and no-face frames [0] are not counted)
- **Per-frame accuracy** (raw CNN, no smoothing): **57.7%**
- **State accuracy** (after smoothing + rules; UNKNOWN counts as wrong): **59.2%**
- UNKNOWN rate: 12.2%
- Time per frame (face crop -> CNN -> state, this machine): mean 27.1 ms, p95 35.8 ms

| true state | frames | per-frame acc | state acc | UNKNOWN |
|---|---|---|---|---|
| alert | 436 | 43.1% | 34.4% | 14.7% |
| microsleep | 185 | 86.5% | 84.9% | 4.9% |
| yawning | 234 | 62.0% | 85.0% | 13.2% |

Confusion matrix (rows: what you acted, columns: what the CNN said, per frame):

| true \ predicted | alert | microsleep | yawning |
|---|---|---|---|
| alert | 188 | 248 | 0 |
| microsleep | 25 | 160 | 0 |
| yawning | 13 | 76 | 145 |

Honest notes: you are the only subject, the labels are what you *meant* to act (a half-closed eye while labelled 'alert' counts as an error), and neighbouring frames are almost identical, so treat this as a demo check, not a final benchmark.