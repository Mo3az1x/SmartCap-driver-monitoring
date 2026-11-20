# Image models: ANN vs CNN

- Data: 53331 images, classes: alert, microsleep, yawning
- Split: 42913 train (80%) / 10418 validation (20%), same split for both models, done BY VIDEO (all frames of a video are on one side only)
- Image size: 64x64, RGB; epochs: 20; batch: 64

| Model | Params | Val acc | Val loss | Train acc | Train loss | Macro F1 | Train time | ms / image |
|---|---|---|---|---|---|---|---|---|
| ANN | 6,441,475 | 89.01% | 0.3247 | 90.02% | 0.1992 | 0.869 | 150 s | 1.43 |
| CNN | 583,267 | 97.49% | 0.1225 | 98.86% | 0.0210 | 0.965 | 648 s | 6.86 |

Val acc = best epoch on the validation set; loss/train columns = last epoch.
Note: video ids are not person ids - the same driver may appear in several videos, so the numbers can still be a little optimistic. Testing on a person / camera the model never saw is the real check.