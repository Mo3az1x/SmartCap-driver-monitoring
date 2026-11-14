# SmartCap deep-learning report

- participants: ['P01', 'P02', 'P03']   windows: 1284   per class: {'attentive': 426, 'distracted': 432, 'drowsy': 426}
- evaluation: subject-independent GroupKFold
- arch: small (9,539 params)   epochs: 15   final ensemble seeds: 1   device: cpu

| model | accuracy | macro-F1 |
|---|---|---|
| 1D-CNN/GRU small (deep learning, 1 seed in CV) | 0.999 | 0.999 |
| RandomForest (baseline) | 0.999 | 0.999 |

DL confusion matrix (rows = true, cols = predicted; order: attentive, distracted, drowsy)
```
[[426   0   0]
 [  0 432   0]
 [  1   0 425]]
```
```
              precision    recall  f1-score   support

   attentive       1.00      1.00      1.00       426
  distracted       1.00      1.00      1.00       432
      drowsy       1.00      1.00      1.00       426

    accuracy                           1.00      1284
   macro avg       1.00      1.00      1.00      1284
weighted avg       1.00      1.00      1.00      1284

```

ONNX inference on this machine's CPU: 0.0 ms per 10 s window (the engine scores 2 windows per second)
Total training time: 0.4 min