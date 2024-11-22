# ECE143 Project

The pre-trained weights for the Ensemble Model and the ExtraTreeClassifier can be downloaded from this Google Drive folder: [Model Weights](https://drive.google.com/drive/folders/1zXDAAJkvUK8wZCzr4DZojyVU2zw_6_uI?usp=sharing). These weights are required for replicating the classification results in our project.

## What We Did in the Project

### **Dataset Collection:**

* Dataset source: [Kaggle - Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification?select=train.csv)

### **Data Preprocessing:**

* Removed unnecessary non-numeric characters.
* Mapped string-form categorical and numerical features to numeric categories (e.g., `"Payment_of_Min_Amount"`, `"Credit_History_Age"`).
* Eliminated noise and outliers in the dataset (e.g., `"Annual_Income"` above the 99% quantile, `"Num_Credit_Card"` above the 97% quantile, etc.).

### **Data Visualization:**

* **Univariate Data Distribution** (e.g., histograms):
  * Examples: Age distribution, Annual Income distribution, etc.
* **Multivariate Correlation Analysis and Distribution** (e.g., heatmaps, boxplots, violin plots):
  * Examples: Annual Income by Credit Score, Credit History Age in Years by Credit Score, Minimum Payment Status vs. Credit Score, etc.

### **Machine Learning Model Training:**

* **Features Used** :
  `Age`, `Occupation`, `Annual_Income`, `Num_Bank_Accounts`, `Num_Credit_Card`, `Num_of_Loan`, `Delay_from_due_date`, `Num_of_Delayed_Payments`, `Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Credit_Mix`, `Outstanding_Debt`, `Payment_of_Min_Amount`, `Total_EMI_per_month`, `Amount_invested_monthly`, `Payment_Behaviour`, `Credit_Score`, `Credit_History_Age_in_Years`.
* Normalized the data using `StandardScaler`, split the dataset into a training set and a test set with a ratio of 9:1, and applied ***SMOTE* **to augment the data after the split.
* ***SMOTE (Synthetic Minority Oversampling Technique)*** was used to balance the dataset by generating synthetic samples for underrepresented classes, ensuring fair model training.
* Trained two models:
  * `b`
  * `b` (combining ***ExtraTree**, **RandomForest***, and ***CatBoost***).

### **Model Performance:**

- ***ExtraTreeClassifier (10 folds):***

|    | Model                          | Accuracy | AUC    | Recall | Prec.  | F1     | Kappa  | MCC    | TT (Sec) |
| -- | ------------------------------ | -------- | ------ | ------ | ------ | ------ | ------ | ------ | -------- |
| et | ***Extra Trees Classifier*** | 0.8527   | 0.9504 | 0.8527 | 0.8516 | 0.8514 | 0.7790 | 0.7797 | 8.2790   |

- ***EnsembleClassifier (5 folds)***:

| Fold | Accuracy | AUC    | Recall | Prec   | F1     | Kappa  | MCC    |
| ---- | -------- | ------ | ------ | ------ | ------ | ------ | ------ |
| 0    | 0.8602   | 0.0000 | 0.8602 | 0.8593 | 0.8595 | 0.7902 | 0.7905 |
| 1    | 0.8561   | 0.0000 | 0.8561 | 0.8551 | 0.8553 | 0.7841 | 0.7844 |
| 2    | 0.8532   | 0.0000 | 0.8532 | 0.8523 | 0.8526 | 0.7798 | 0.7799 |
| 3    | 0.8530   | 0.0000 | 0.8530 | 0.8520 | 0.8522 | 0.7795 | 0.7797 |
| 4    | 0.8562   | 0.0000 | 0.8562 | 0.8555 | 0.8558 | 0.7843 | 0.7843 |
| Mean | 0.8557   | 0.0000 | 0.8557 | 0.8548 | 0.8551 | 0.7836 | 0.7838 |
| Std  | 0.0026   | 0.0000 | 0.0026 | 0.0026 | 0.0026 | 0.0039 | 0.0039 |

The ***StackingClassifier* **showed slightly better performance in terms of *Accuracy* and *F1 scores* compared to the ***ExtraTreeClassifier***.

### Feature Optimization

Determine the search space based on user input values and professional recommendations, combined with the correlation between features and the Credit Score. For example:

*'Total_EMI_per_month': (0, constraints["max_emi_to_income"] * current_sample["Annual_Income"] / 12)*

*"Delay_from_due_date": (0, current_sample["Delay_from_due_date"])*

Features are added to the search space in order of their importance as determined by the model. ***Bayesian optimization*** is then used to perform reverse optimization within this search space.

### Note

The ***_old models*** in Google Drive were trained using only a subset of features and are therefore not suitable for the current optimization process.

In the current optimization process, it is possible for values to exceed the bounds defined in `feature_bounds`. This occurs because `feature_bounds` provides a one-sided constraint (either an upper or lower bound) based on the correlation between the current feature and `Credit_score`. The other boundary is determined by the user-inputted current value.

For example, if `Num_of_Loan` is negatively correlated with `Credit_Score`, and the user inputs `Num_of_Loan` as 25 (far exceeding our defined upper bound for the search), the search range would then be `(feature_bounds["Num_of_Loan"][0], 25)`.
