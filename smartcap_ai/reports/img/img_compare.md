# Image models: ANN vs CNN

- Data: 53331 images, classes: alert, microsleep, yawning
- Split: 42913 train (80%) / 10418 validation (20%), same split for both models, done BY VIDEO (all frames of a video are on one side only)
- Image size: 64x64, RGB; epochs: 20; batch: 32

| Model | Params | Val acc | Val loss | Train acc | Train loss | Macro F1 | Train time | ms / image |
|---|---|---|---|---|---|---|---|---|
| CNN | 583,267 | 97.83% | 0.0788 | 98.77% | 0.0225 | 0.971 | 346 s | 0.79 |

Val acc = best epoch on the validation set; loss/train columns = last epoch.
Note: video ids are not person ids - the same driver may appear in several videos, so the numbers can still be a little optimistic. Testing on a person / camera the model never saw is the real check.