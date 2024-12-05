import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from pycaret.classification import *
import pycaret
import joblib
import argparse
import kagglehub

parser = argparse.ArgumentParser(description='Credit Scoring Model Training')
parser.add_argument("--ensemble", type=bool, default=False, help="Train an ensemble model or not")

args = parser.parse_args()
use_ensemble = args.ensemble

print("=" * 20 + "Downloading dataset..." + "=" * 20)
path = kagglehub.dataset_download("parisrohan/credit-score-classification")
print("Path to dataset files:", path)

label_encoder = LabelEncoder()

print("=" * 20 + "Loading data..." + "=" * 20)

df = pd.read_csv(f"{path}\\train.csv")

def remove_underscore(df, col):
    df[col] = df[col].apply(lambda x: str(x).replace("_", "") if str(x) else x)
    df[col] = pd.to_numeric(df[col], errors="coerce")

def convert_to_year_fraction(credit_history_age):
    if pd.isnull(credit_history_age):
        return np.nan
    match = re.match(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", credit_history_age)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months / 12
    return np.nan

print("=" * 20 + "Cleaning dataset..." + "=" * 20)

remove_underscore(df, "Age")
remove_underscore(df, "Annual_Income")
remove_underscore(df, "Outstanding_Debt")
remove_underscore(df, "Monthly_Balance")
remove_underscore(df, "Num_of_Loan")
remove_underscore(df, "Num_of_Delayed_Payment")
remove_underscore(df, "Amount_invested_monthly")
remove_underscore(df, "Credit_Mix")
remove_underscore(df, "Changed_Credit_Limit")

df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].str.strip().str.lower()
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'yes': 1, 'no': 0})
df["Credit_Score"] = df["Credit_Score"].map({"Good":2, "Poor":0, "Standard":1})

df["Occupation"] = label_encoder.fit_transform(df["Occupation"])
df["Credit_Mix"] = label_encoder.fit_transform(df["Credit_Mix"])
df['Credit_History_Age_in_Years'] = df['Credit_History_Age'].apply(convert_to_year_fraction)

print("==================================================================")
# Step 1: Remove the top 1% of rows with the highest annual income.
quantiles = df['Annual_Income'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.998])
print("Annual Income Quantiles:")
print(quantiles)
q99 = quantiles[0.99]
df_filtered = df[df['Annual_Income'] <= q99].copy()
print(f"Rows remaining after filtering top 1% on annual income: {len(df_filtered)}")
print("==================================================================")

# Step 2: Remove the top 2% of rows with the most bank accounts.
quantiles = df_filtered['Num_Bank_Accounts'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995])
print("Number of Bank Accounts Quantiles:")
print(quantiles)
q98 = quantiles[0.98]
df_filtered = df_filtered[(df_filtered['Num_Bank_Accounts'] <= q98) & (df_filtered["Num_Bank_Accounts"] >= 0)]
print(f"Rows remaining after filtering top 2% on number of bank accounts: {len(df_filtered)}")
print("==================================================================")

# Step 3: Remove the top 3% of rows with the highest number of credit cards.
quantiles = df_filtered['Num_Credit_Card'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995])
print("Number of Credit Card Quantiles:")
print(quantiles)
q97 = quantiles[0.97]
df_filtered = df_filtered[(df_filtered['Num_Credit_Card'] <= q97) & (df_filtered["Num_Credit_Card"] >= 0)]
print(f"Rows remaining after filtering top 3% on number of credit cards: {len(df_filtered)}")
print("==================================================================")

# Step 4: Remove the top 2% of rows with the most credit inquiries.
quantiles = df_filtered['Num_Credit_Inquiries'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995])
print("Number of Credit Inquiries Quantiles:")
print(quantiles)
q98 = quantiles[0.98]
df_filtered = df_filtered[(df_filtered['Num_Credit_Inquiries'] <= q98) & (df_filtered["Num_Credit_Inquiries"] >= 0)]
print(f"Rows remaining after filtering top 2% on number of credit inquiries: {len(df_filtered)}")
print("==================================================================")

# Step 5: Remove the top 0.47% of rows with the most loans.
quantiles = df_filtered['Num_of_Loan'].quantile([0.25, 0.50, 0.75, 0.90, 0.97, 0.98, 0.99, 0.995, 0.9951, 0.9953, 0.9954, 0.996, 0.998])
print("Number of Loans Quantiles:")
print(quantiles)
q9953 = quantiles[0.9953]
df_filtered = df_filtered[(df_filtered['Num_of_Loan'] <= q9953) & (df_filtered["Num_of_Loan"] >= 0)]
print(f"Rows remaining after filtering top 0.47% on number of loans: {len(df_filtered)}")
print("==================================================================")

# Step 6: Remove the top 0.8% of rows with the highest number of delayed payments.
quantiles = df_filtered['Num_of_Delayed_Payment'].quantile([0.25, 0.50, 0.75, 0.90, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.995])
print("Number of Delayed Payments Quantiles:")
print(quantiles)
q992 = quantiles[0.992]
df_filtered = df_filtered[(df_filtered['Num_of_Delayed_Payment'] <= q992) & (df_filtered["Num_of_Delayed_Payment"] >= 0)]
print(f"Rows remaining after filtering top 0.8% on number of delayed payments: {len(df_filtered)}")
print("==================================================================")

# Step 7: Filter out rows with negative monthly balance values.
quantiles = df_filtered['Monthly_Balance'].quantile([0.25, 0.50, 0.75, 0.90, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.995, 0.998, 0.9999])
print("Monthly Balance Quantiles:")
print(quantiles)
df_filtered = df_filtered[(df_filtered["Monthly_Balance"] >= 0)]
print(f"Rows remaining after filtering on monthly balance: {len(df_filtered)}")
print("==================================================================")

quantiles = df_filtered['Amount_invested_monthly'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.952, 0.954, 0.956, 0.958, 0.96])
print("Amount_invested_monthly Quantiles:")
print(quantiles)
q954 = quantiles[0.954]
df_filtered = df_filtered[(df_filtered['Amount_invested_monthly'] <= q954) & (df_filtered["Amount_invested_monthly"] >= 0)]
print(f"Rows remaining after filtering top 3% on Amount_invested_monthly: {len(df_filtered)}")
print("==================================================================")

quantiles = df_filtered['Total_EMI_per_month'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.96, 0.965, 0.968, 0.99, 0.995])
print("Total_EMI_per_month Quantiles:")
print(quantiles)
q965 = quantiles[0.965]
df_filtered = df_filtered[(df_filtered['Total_EMI_per_month'] <= q965) & (df_filtered["Total_EMI_per_month"] >= 0)]
print(f"Rows remaining after filtering top 3% on Total_EMI_per_month: {len(df_filtered)}")
print("==================================================================")

# Step 8: Handle abnormal values in age by setting out-of-range ages to NaN and interpolating.
df_filtered.loc[(df_filtered['Age'] < 0) | (df_filtered['Age'] > 100), 'Age'] = pd.NA
df_filtered['Age'] = df_filtered.groupby('Name')['Age'].transform(lambda x: x.interpolate(method='linear'))
print("Handled abnormal age values with interpolation where possible.")
print("==================================================================")

# Step 9: Fill missing values for Monthly Inhand Salary using linear interpolation.
df_filtered['Monthly_Inhand_Salary'] = df_filtered.groupby('Name')['Monthly_Inhand_Salary'].transform(lambda x: x.interpolate(method='linear'))

# Step 10: Drop rows with missing values in Age, Outstanding Debt, and Monthly Balance.
df_filtered.dropna(subset=["Age", "Outstanding_Debt", "Monthly_Balance"], inplace=True)
print(f"Rows remaining after dropping rows with missing values in critical fields: {len(df_filtered)}")
print("==================================================================")

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

s = setup(X_train_sm, target=y_train_sm, normalize=True, n_jobs=-1, use_gpu=True, preprocess=False)

print("=" * 20 + "Training..." + "=" * 20)
if use_ensemble:
    model1 = create_model('rf')
    model2 = create_model('catboost')
    model3 = create_model('et')
    sklearn_model = stack_models([model1, model2, model3], fold=5)
    evaluate_model(sklearn_model)
    joblib.dump(sklearn_model, 'EnsembleClf_sklearn.pkl')
else:
    sklearn_model = compare_models(include=['et', 'rf', 'catboost'])
    plot_model(sklearn_model, plot = 'confusion_matrix')
    save_model(sklearn_model, 'best_model')
    joblib.dump(sklearn_model, 'ExtraTreeClf_sklearn.pkl')