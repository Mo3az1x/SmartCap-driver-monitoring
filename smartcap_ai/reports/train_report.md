# SmartCap training report

- recordings: 4   participants: ['P01', 'P02', 'P03', 'P04']
- windows (10 s, 1 s stride): 1712   per class: {'distracted': 576, 'attentive': 568, 'drowsy': 568}

## Evaluation: subject-independent GroupKFold
- fold 1: accuracy 1.000 (test participants/blocks: [np.str_('P04')])
- fold 2: accuracy 0.998 (test participants/blocks: [np.str_('P03')])
- fold 3: accuracy 1.000 (test participants/blocks: [np.str_('P02')])
- fold 4: accuracy 1.000 (test participants/blocks: [np.str_('P01')])

**accuracy 0.999   macro-F1 0.999**

```
              precision    recall  f1-score   support

   attentive       1.00      1.00      1.00       568
  distracted       1.00      1.00      1.00       576
      drowsy       1.00      1.00      1.00       568

    accuracy                           1.00      1712
   macro avg       1.00      1.00      1.00      1712
weighted avg       1.00      1.00      1.00      1712

```

Confusion matrix (rows = true, cols = predicted; order: attentive, distracted, drowsy)
```
[[568   0   0]
 [  0 576   0]
 [  1   0 567]]
```

## Most useful features

- yaw_abs_mean: 0.121
- yaw_std: 0.120
- longest_turn_s: 0.112
- turned_frac: 0.109
- ear_norm_min: 0.098
- ear_norm_std: 0.086
- longest_closure_s: 0.079
- perclos: 0.061
- ear_norm_p10: 0.054
- ear_norm_mean: 0.051