import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import argparse
import kagglehub
from hyperopt import fmin, tpe, hp, Trials
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier

parser = argparse.ArgumentParser(description='Credit Scoring Model Training')
parser.add_argument("--model_path", type=str, default="", help="pretrained model path")
parser.add_argument("--data_index", type=int, default=20, help="sample data index")

args = parser.parse_args()
model_path = args.model_path
data_index = args.data_index

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

df_numeric = df_filtered.select_dtypes(include=['float64', 'int64'])
correlation_matrix = df_numeric.corr()

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

best_model = joblib.load(model_path)

X_test_original = X_test.copy()
X_test_original = pd.DataFrame(sc.inverse_transform(X_test_original), columns=X_test_original.columns)
X_train_original = X_train.copy()
X_train_original = pd.DataFrame(sc.inverse_transform(X_train_original), columns=X_train_original.columns)

assert data_index < len(X_test_original), f"data index out of range. Please limit the index between 0 to {len(X_test_original) - 1}."

current_sample = X_test_original.iloc[data_index]

class CreditScoreOptimizer:
    def __init__(self, current_sample, X_train, sc, best_model):
        """
            Parameters:
            current_sample: samples that are not normalized
            X_train: training data that are not normalized
            sc: StandardScaler
            best_model: trained model
        """
        self.current_sample = current_sample
        self.X_train = X_train
        self.sc = sc
        self.best_model = best_model
        self.max_values = X_train.max()
        self.min_values = X_train.min()
        self.unchanged_features = ["Age", "Occupation", "Payment_Behaviour", "Credit_History_Age_in_Years", "Interest_Rate"]

        self.constraints = {
            'max_dti_ratio': 0.43,      # Maximum debt-to-income ratio
            'max_utilization': 0.80,    # Maximum credit utilization
            'max_emi_to_income': 0.36   # Maximum EMI to income ratio
        }

        self.categorical_values = {feature: sorted(X_train[feature].unique()) for feature in ['Credit_Mix', 'Payment_Behaviour'] if feature in X_train.columns}

        self.feature_types = {
            'integer_features': ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries'],
            'binary_features': ['Payment_of_Min_Amount'],
            'categorical_features': list(self.categorical_values.keys()),
            'continuous_features': ['Annual_Income', 'Changed_Credit_Limit', 'Outstanding_Debt', 'Total_EMI_per_month',
                                  'Amount_invested_monthly', 'Credit_History_Age_in_Years', 'Delay_from_due_date']
        }

        self.feature_relationships = {
            'Total_EMI_per_month': {
                'formula': lambda params: (min((params['Annual_Income']/12) * self.constraints['max_emi_to_income'], params['Total_EMI_per_month'])),
                'weight': 1.0
            },
            'Amount_invested_monthly': {
                'formula': lambda params: (min((params['Annual_Income']/12 - params['Total_EMI_per_month']) * 0.7, params['Amount_invested_monthly'])),
                'weight': 1.0
            }
        }

        # Reasonable feature bounds
        self.feature_bounds = {
            'Num_Bank_Accounts': (1, 6),
            'Num_Credit_Card': (1, 5),
            'Num_of_Loan': (1, 4),
            "Delay_from_due_date": (0, 30),
            "Num_of_Delayed_Payment": (0, 9),
            "Changed_Credit_Limit": (0, 8),
            "Num_Credit_Inquiries": (0, 8),
            "Outstanding_Debt": (0.1 * self.current_sample["Outstanding_Debt"], self.constraints["max_dti_ratio"] * self.current_sample["Outstanding_Debt"]),
            "Annual_Income": (0.95 * self.current_sample["Annual_Income"], 1.05 * self.current_sample["Annual_Income"]),
            'Total_EMI_per_month': (0, self.constraints["max_emi_to_income"] * self.current_sample["Annual_Income"] / 12),
            'Amount_invested_monthly': (0, 0.4 * self.current_sample["Annual_Income"] / 12)
        }

    def get_feature_importance(self):
        """Feature Importance Ranking"""
        if isinstance(self.best_model, ExtraTreesClassifier):
            importance = self.best_model.feature_importances_
        elif isinstance(self.best_model, StackingClassifier):
            importance = self.best_model.estimators_[2].feature_importances_

        feature_importance = pd.DataFrame({'feature': self.X_train.columns, 'importance': importance})

        # Exclude unused features
        feature_importance = feature_importance[~feature_importance['feature'].isin(['Age', 'Occupation'])]

        return feature_importance.sort_values('importance', ascending=False)

    def _normalize_sample(self, sample):
        """Normalize samples"""
        sample_df = pd.DataFrame([sample])
        return pd.DataFrame(self.sc.transform(sample_df), columns=sample_df.columns)

    def _check_constraints(self, params):
        """Calculate penalty for constraint violations"""
        penalty = 0

        # Check feature relationship constraints
        for feature, relationship in self.feature_relationships.items():
            if feature in params:
                expected_value = relationship['formula'](params)
                actual_value = params[feature]
                # print("!", feature, expected_value, actual_value)
                penalty += abs(expected_value - actual_value) * relationship['weight']

        # Check global constraints
        if 'Total_EMI_per_month' in params and 'Monthly_Inhand_Salary' in params:
            if params['Total_EMI_per_month'] / params['Monthly_Inhand_Salary'] > self.constraints['max_emi_to_income']:
                print("Max emi to income constraints exceeded.")
                penalty += 1000
        if 'Credit_Utilization_Ratio' in params and params['Credit_Utilization_Ratio'] > self.constraints['max_utilization']:
            print("Max utilization ration constraints exceeded.")
            penalty += 1000
        if 'Monthly_Balance' in params and params['Monthly_Balance'] < self.constraints['min_monthly_balance']:
            print("Min monthly balance exceeded.")
            penalty += 1000

        return penalty

    def _get_feature_space(self, feature, correlation, epsilon=1e-4):
        """Define search space"""
        current_value = self.current_sample[feature]
        if feature in self.feature_types['categorical_features']:
            return hp.choice(feature, self.categorical_values[feature])
        if feature in self.feature_types['binary_features']:
            return hp.choice(feature, [0, 1])

        # Get boundary
        bounds = self.feature_bounds.get(feature, (None, None))
        min_bound = bounds[0] if bounds[0] is not None else float('-inf')
        max_bound = bounds[1] if bounds[1] is not None else float('inf')

        # Define search bound
        if correlation > 0:
            lower = current_value
            upper = min(max_bound, max(self.max_values[feature], current_value + epsilon))
        else:
            lower = max(min_bound, min(self.min_values[feature], current_value - epsilon))
            upper = current_value
        if feature in self.feature_types['integer_features']:
            return hp.quniform(feature, lower, upper, 1)
        return hp.uniform(feature, lower, upper)
    
    def get_current_pred(self, current_input):
        normalized_sample = self._normalize_sample(current_input)
        pred = self.best_model.predict(normalized_sample)[0]
        return pred

    def progressive_optimize(self, correlation_matrix, max_features=None, base_max_evals=100):
        """
          Progressively adding features into the optimization space

          Parameters:
              max_features : int or None
              base_max_evals : int
        """
        feature_importance = self.get_feature_importance()
        if max_features is not None:
            feature_importance = feature_importance.head(max_features)
        best_score = float('-inf')
        best_params = None
        current_best_sample = self.current_sample.copy()

        for i in range(len(feature_importance)):
            current_features = feature_importance['feature'].iloc[:i+1].tolist()
            print(f"\nOptimizing with features: {current_features}")

            space = {}
            skip_flag = False
            for j, feature in enumerate(current_features):
                if feature not in correlation_matrix["Credit_Score"]:
                    print(f"Feature {feature} not found in correlation_matrix")
                    if j == len(current_features) - 1:
                      skip_flag = True
                    continue
                elif feature in self.unchanged_features:
                    print(f"Feature {feature} will not be included in the search space.")
                    if j == len(current_features) - 1:
                      skip_flag = True
                    continue
                correlation = correlation_matrix["Credit_Score"][feature]
                space[feature] = self._get_feature_space(feature, correlation)
            if skip_flag:
                continue
            current_max_evals = base_max_evals * (i + 1)

            def objective(params):
                modified_sample = current_best_sample.copy()
                for feature, value in params.items():
                    if feature in self.feature_types['categorical_features']:
                        choices = self.categorical_values[feature]
                        modified_sample[feature] = choices[int(value)]
                    else:
                        modified_sample[feature] = value

                # Add constraint penalty
                constraint_penalty = self._check_constraints(modified_sample)

                # Normalize samples
                normalized_sample = self._normalize_sample(modified_sample)
                pred = self.best_model.predict(normalized_sample)[0]

                return 2 - (pred - constraint_penalty * 0.1)

            trials = Trials()
            best = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=current_max_evals,
                trials=trials
            )

            # Update current sample
            for feature, value in best.items():
                if feature in self.feature_types['categorical_features']:
                    choices = self.categorical_values[feature]
                    current_best_sample[feature] = choices[int(value)]
                elif feature in self.feature_types['integer_features']:
                    current_best_sample[feature] = int(round(value))
                else:
                    current_best_sample[feature] = value

            # Evaluate current result
            normalized_best = self._normalize_sample(current_best_sample)
            current_score = self.best_model.predict(normalized_best)[0]
            print(current_best_sample[current_features])

            print(f"Current score with {i+1} features: {current_score}")

            # Update the global best score
            if current_score > best_score:
                best_score = current_score
                best_params = current_best_sample.copy()
                print("Found new best score!")

            if best_score == 2:
                print("Best parameters found:")
                print(current_best_sample[current_features])
                print("Reached target score, stopping optimization.")
                break

        return best_params, best_score

optimizer = CreditScoreOptimizer(current_sample=current_sample, X_train=X_train_original, sc=sc, best_model=best_model)

optimized_params = optimizer.progressive_optimize(correlation_matrix, base_max_evals=10)
print("Before optimization: ")
print(current_sample)
print("After optimization: ")
print(optimized_params)