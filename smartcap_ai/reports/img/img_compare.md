# Image models: ANN vs CNN

- Data: 53331 images, classes: alert, microsleep, yawning
- Split: 42913 train (80%) / 10418 validation (20%), stratified, same split for both models
- Image size: 64x64, RGB; epochs: 20; batch: 64

| Model | Params | Val acc | Val loss | Train acc | Train loss | Macro F1 | Train time | ms / image |
|---|---|---|---|---|---|---|---|---|
| ANN | 6,441,475 | 89.01% | 0.3247 | 90.02% | 0.1992 | 0.869 | 183 s | 2.68 |
| CNN | 583,267 | 97.27% | 0.1021 | 98.85% | 0.0209 | 0.961 | 852 s | 2.52 |

Val acc = best epoch on the validation set; loss/train columns = last epoch.
**Honest note:** the split is random per image. If frames of the same person appear in both train and validation, the numbers are optimistic. A test on a person/camera the model never saw (your own images) is the real check.