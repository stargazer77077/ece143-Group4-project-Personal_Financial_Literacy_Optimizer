import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pycaret.classification import *
import pycaret
import joblib
import argparse
from utils import data_process

parser = argparse.ArgumentParser(description='Credit Scoring Model Training')
parser.add_argument("--ensemble", type=str, default="False", help="Train an ensemble model or not")
parser.add_argument("--use_gpu", type=str, default="True", help="use GPU for training")
parser.add_argument("--data_path", type=str, default="", help="data path")
args = parser.parse_args()
use_ensemble = args.ensemble == "True"
use_gpu = args.use_gpu == "True"
data_path = args.data_path
data_path = None if data_path == "" else data_path

df_filtered = data_process(data_path)

payments_dict = {
    'High_spent_Small_value_payments' : 0,
    'Low_spent_Large_value_payments' : 1,
    'Low_spent_Medium_value_payments' : 2,
    'Low_spent_Small_value_payments' : 3,
    'High_spent_Medium_value_payments' : 4,
    'High_spent_Large_value_payments': 5,
    '!@9#%8' : np.nan
}

df_filtered['Payment_Behaviour'] = df_filtered['Payment_Behaviour'].map(payments_dict)

df_filtered.drop("ID", axis=1, inplace=True)
df_filtered.drop("Name", axis=1, inplace=True)
df_filtered.drop("Customer_ID", axis=1, inplace=True)
df_filtered.drop("SSN", axis=1, inplace=True)
df_filtered.drop("Type_of_Loan", axis=1, inplace=True)
df_filtered.drop("Monthly_Inhand_Salary", axis=1, inplace=True)
df_filtered.drop("Credit_History_Age", axis=1, inplace=True)
df_filtered.drop("Month", axis=1, inplace=True)
df_filtered.drop(['Monthly_Balance', 'Credit_Utilization_Ratio', "Interest_Rate", "Occupation", "Age"], axis=1, inplace=True)

df_filtered = df_filtered.dropna()

X = df_filtered.drop(['Credit_Score'], axis=1)
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
y = df_filtered['Credit_Score']
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
sm = SMOTE(k_neighbors=7)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

s = setup(X_train_sm, target=y_train_sm, normalize=True, n_jobs=-1, use_gpu=use_gpu, preprocess=False)

print("=" * 20 + "Training..." + "=" * 20)
if use_ensemble:
    model1 = create_model('rf')
    model2 = create_model('catboost')
    model3 = create_model('et')
    sklearn_model = stack_models([model1, model2, model3], fold=5)
    evaluate_model(sklearn_model)
    joblib.dump(sklearn_model, 'EnsembleClf_sklearn.pkl')
else:
    sklearn_model = compare_models(include=['et', 'rf', "catboost"])
    plot_model(sklearn_model, plot = 'confusion_matrix')
    save_model(sklearn_model, 'best_model')
    joblib.dump(sklearn_model, 'ExtraTreeClf_sklearn.pkl')