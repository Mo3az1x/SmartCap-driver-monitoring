# Lighting / framing experiment

Classes: ['alert', 'microsleep', 'yawning']. Recall = per-class recall; `micro` = microsleep (eyes closed).

## Part 1 - sensitivity on FL3D validation images (known-good images, one change at a time)

| Condition | ANN acc | ANN micro recall | CNN acc | CNN micro recall |
|---|---|---|---|---|
| none | 90.1% | 86.7% | 98.0% | 94.8% |
| darker x0.4 | 90.0% | 86.7% | 97.9% | 94.6% |
| darker x0.7 | 90.1% | 86.7% | 97.9% | 94.4% |
| brighter x1.5 | 90.1% | 86.7% | 97.9% | 94.4% |
| blue colour cast | 87.6% | 87.1% | 97.7% | 97.4% |
| side light (gradient) | 89.9% | 76.2% | 96.4% | 88.9% |
| shift 4 px | 73.0% | 84.3% | 96.7% | 85.3% |
| zoom in (crop 85%) | 86.1% | 91.1% | 98.2% | 97.2% |
| zoom out (face 85%) | 85.5% | 87.3% | 90.5% | 51.8% |

## Part 2 - extra pre-processing before the model, on YOUR webcam images

| Condition | ANN acc | ANN micro recall | CNN acc | CNN micro recall |
|---|---|---|---|---|
| none (current) | 63.9% | 0.0% | 97.0% | 96.1% |
| gray-world (per-channel) | 64.3% | 1.0% | 97.0% | 96.1% |
| CLAHE contrast | 63.0% | 0.0% | 91.1% | 79.4% |
| auto-gamma (mean 0.5) | 63.9% | 0.0% | 97.0% | 96.1% |
| gray-world + CLAHE | 63.6% | 0.0% | 90.5% | 77.5% |

## Part 2b - framing change on YOUR webcam images

| Condition | ANN acc | ANN micro recall | CNN acc | CNN micro recall |
|---|---|---|---|---|
| none | 63.9% | 0.0% | 97.0% | 96.1% |
| zoom in (crop 85%) | 34.4% | 0.0% | 96.1% | 98.0% |
| zoom out (face 85%) | 63.6% | 0.0% | 66.9% | 2.9% |
| shift 4 px | 61.6% | 1.0% | 96.7% | 94.1% |

## Part 3 - what the models think of YOUR microsleep images

| Model | mean P(alert) | mean P(microsleep) | mean P(yawning) |
|---|---|---|---|
| ANN | 0.878 | 0.052 | 0.070 |
| CNN | 0.038 | 0.962 | 0.000 |
