# Image models: ANN vs CNN

- Data: 56381 images, classes: alert, microsleep, yawning
- Split: 45963 train (80%) / 10418 validation (20%), same split for both models, done BY VIDEO (all frames of a video are on one side only)
- Image size: 64x64, RGB; epochs: 20; batch: 64

| Model | Params | Val acc | Val loss | Train acc | Train loss | Macro F1 | Train time | ms / image |
|---|---|---|---|---|---|---|---|---|
| ANN | 6,441,475 | 90.06% | 0.3014 | 90.28% | 0.1964 | 0.882 | 2392 s | 3.50 |
| CNN | 583,267 | 97.93% | 0.0643 | 98.83% | 0.0210 | 0.973 | 52193 s | 7.01 |

Val acc = best epoch on the validation set; loss/train columns = last epoch.
Note: video ids are not person ids - the same driver may appear in several videos, so the numbers can still be a little optimistic. Testing on a person / camera the model never saw is the real check.

## Holdout person (never used in training)

| Model | Accuracy |
|---|---|
| ANN | 81.29% |
| CNN | 75.44% |
