# ECE143 Project

The pre-trained weights for the Ensemble Model and the ExtraTreeClassifier can be downloaded from this Google Drive folder: [Model Weights](https://drive.google.com/drive/folders/1zXDAAJkvUK8wZCzr4DZojyVU2zw_6_uI?usp=sharing). These weights are required for replicating the classification results in our project.

### Model performance:

ExtraTreeClassifier: 

|    | Model                  | Accuracy | AUC    | Recall | Prec.  | F1     | Kappa  | MCC    | TT (Sec) |
| -- | ---------------------- | -------- | ------ | ------ | ------ | ------ | ------ | ------ | -------- |
| et | Extra Trees Classifier | 0.8527   | 0.9504 | 0.8527 | 0.8516 | 0.8514 | 0.7790 | 0.7797 | 8.2790   |

EnsembleClassifier:

| Fold | Accuracy | AUC    | Recall | Prec   | F1     | Kappa  | MCC    |
| ---- | -------- | ------ | ------ | ------ | ------ | ------ | ------ |
| 0    | 0.8602   | 0.0000 | 0.8602 | 0.8593 | 0.8595 | 0.7902 | 0.7905 |
| 1    | 0.8561   | 0.0000 | 0.8561 | 0.8551 | 0.8553 | 0.7841 | 0.7844 |
| 2    | 0.8532   | 0.0000 | 0.8532 | 0.8523 | 0.8526 | 0.7798 | 0.7799 |
| 3    | 0.8530   | 0.0000 | 0.8530 | 0.8520 | 0.8522 | 0.7795 | 0.7797 |
| 4    | 0.8562   | 0.0000 | 0.8562 | 0.8555 | 0.8558 | 0.7843 | 0.7843 |
| Mean | 0.8557   | 0.0000 | 0.8557 | 0.8548 | 0.8551 | 0.7836 | 0.7838 |
| Std  | 0.0026   | 0.0000 | 0.0026 | 0.0026 | 0.0026 | 0.0039 | 0.0039 |

### Note

The ***_old models*** in Google Drive were trained using only a subset of features and are therefore not suitable for the current optimization process.

In the current optimization process, it is possible for values to exceed the bounds defined in `self.feature_bounds`. This occurs because `self.feature_bounds` provides a one-sided constraint (either an upper or lower bound) based on the correlation between the current feature and `Credit_score`. The other boundary is determined by the user-inputted current value.

For example, if `Num_of_Loan` is negatively correlated with `Credit_Score`, and the user inputs `Num_of_Loan` as 25 (far exceeding our defined upper bound for the search), the search range would then be `(self.feature_bounds["Num_of_Loan"][0], 25)`.
