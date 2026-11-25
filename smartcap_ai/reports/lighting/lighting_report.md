# Lighting / framing experiment

Classes: ['alert', 'microsleep', 'yawning']. Recall = per-class recall; `micro` = microsleep (eyes closed).

## Part 1 - sensitivity on FL3D validation images (known-good images, one change at a time)

| Condition | ANN acc | ANN micro recall | CNN acc | CNN micro recall |
|---|---|---|---|---|
| none | 90.1% | 83.5% | 97.9% | 93.8% |
| darker x0.4 | 90.1% | 83.5% | 97.9% | 93.5% |
| darker x0.7 | 90.1% | 83.5% | 97.7% | 92.7% |
| brighter x1.5 | 90.0% | 83.5% | 97.9% | 93.5% |
| blue colour cast | 89.3% | 78.4% | 97.5% | 92.5% |
| side light (gradient) | 86.8% | 50.2% | 97.2% | 91.1% |
| shift 4 px | 85.2% | 75.2% | 98.2% | 91.9% |
| zoom in (crop 85%) | 75.6% | 97.2% | 98.2% | 95.8% |
| zoom out (face 85%) | 84.0% | 66.7% | 95.4% | 75.6% |

## Part 2 - extra pre-processing before the model, on YOUR webcam images

| Condition | ANN acc | ANN micro recall | CNN acc | CNN micro recall |
|---|---|---|---|---|
| none (current) | 64.6% | 2.9% | 94.4% | 92.2% |
| gray-world (per-channel) | 64.3% | 2.9% | 93.8% | 90.2% |
| CLAHE contrast | 38.0% | 0.0% | 90.5% | 77.5% |
| auto-gamma (mean 0.5) | 64.3% | 1.0% | 94.4% | 92.2% |
| gray-world + CLAHE | 39.0% | 0.0% | 90.2% | 77.5% |

## Part 2b - framing change on YOUR webcam images

| Condition | ANN acc | ANN micro recall | CNN acc | CNN micro recall |
|---|---|---|---|---|
| none | 64.6% | 2.9% | 94.4% | 92.2% |
| zoom in (crop 85%) | 34.1% | 4.9% | 94.4% | 95.1% |
| zoom out (face 85%) | 46.9% | 1.0% | 77.7% | 40.2% |
| shift 4 px | 59.7% | 12.7% | 93.1% | 90.2% |

## Part 3 - what the models think of YOUR microsleep images

| Model | mean P(alert) | mean P(microsleep) | mean P(yawning) |
|---|---|---|---|
| ANN | 0.607 | 0.306 | 0.087 |
| CNN | 0.089 | 0.910 | 0.000 |
