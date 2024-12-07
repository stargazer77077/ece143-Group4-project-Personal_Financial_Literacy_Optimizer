import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from hyperopt import fmin, tpe, hp, Trials
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier

def remove_underscore(df, col):
    '''
    Function to remove underscore from dataframe column names.
    '''
    df[col] = df[col].apply(lambda x: str(x).replace("_", "") if str(x) else x)
    df[col] = pd.to_numeric(df[col], errors="coerce")

def convert_to_year_fraction(credit_history_age):
    '''
    Function to convert length of time into year fraction.
    '''
    if pd.isnull(credit_history_age):
        return np.nan
    match = re.match(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", credit_history_age)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months / 12
    return np.nan

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

    def get_current_pred(self, current_input):
        '''
        Get model predictions based on output.
        '''
        normalized_sample = self._normalize_sample(current_input)
        pred = self.best_model.predict(normalized_sample)[0]
        if pred == 0:
            return "Poor"
        elif pred == 1:
            return "Standard"
        elif pred == 2:
            return "Good"

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
                '''
                Define objective function.
                '''
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
    def generate_advice(self, optimized_params, original_params):
        """
        Generates advice by comparing optimized parameters with original parameters.

        Parameters:
            optimized_params (dict): Optimized feature values.
            original_params (pd.Series -> dict): Original feature values before optimization.

        Returns:
            dict: A dictionary with features as keys and advice strings as values.
        """
        if isinstance(original_params, pd.Series):
            original_params = original_params.to_dict()  # Convert Series to dict

        advice = {}
        for feature, optimized_value in optimized_params.items():
            original_value = original_params.get(feature, None)
            if original_value is None:
                advice[feature] = f"No data available for comparison for {feature}."
                continue

          # General Features
            if feature in self.feature_bounds:
                min_bound, max_bound = self.feature_bounds[feature]

                if optimized_value != original_value:
                    if feature in self.feature_types["continuous_features"]:
                        advice[feature] = (
                            f"We suggest that you change {feature} from {original_value:.2f} to {optimized_value:.2f} "
                            f"to optimize your financial health. If possible, keep {feature} within the range of ({min_bound:.2f}, {max_bound:.2f})."
                        )
                    elif feature in self.feature_types["integer_features"]:
                        advice[feature] = (
                            f"We suggest that you change {feature.replace('_', ' ')} from {int(original_value)} to {int(optimized_value)}. "
                            f"This change aligns better with your financial goals."
                        )
                    elif feature in self.feature_types["binary_features"]:
                        advice[feature] = (
                            f"We suggset that you switch {feature.replace('_', ' ')} from {int(original_value)} to {int(optimized_value)}. "
                            f"This helps improve your financial standing."
                        )
                    elif feature in self.feature_types["categorical_features"]:
                        advice[feature] = (
                            f"We suggest that you switch {feature} from {original_value} to {optimized_value}. "
                            f"This adjustment is recommended for better outcomes."
                        )
                # else:
                #     advice[feature] = (
                #         f"Your current {feature} at {optimized_value} is optimal for your financial wellbeing. "
                #     )

        return advice
    
def initialize_advice_dict(features):
    """
    Initializes an advice dictionary with each feature as a key and an empty list.
    """
    return {feature: [] for feature in features}

def append_resources(advice_dict):
    """
    Appends resource links to the advice for each feature in the dictionary.

    Parameters:
        advice_dict (dict): The original advice dictionary with feature-specific advice.

    Returns:
        dict: The updated advice dictionary with resource links appended.
    """
    advice_resources = {'Age': ["https://www.nerdwallet.com/article/finance/what-is-the-average-credit-score-by-age-and-what-is-a-good-score-for-my-age"],
                    'Occupation': ["https://www.experian.com/blogs/ask-experian/credit-education/life-events/employment/"],
                'Annual_Income': ["https://www.cnbc.com/select/how-does-salary-and-income-impact-your-credit-score/"],
                'Monthly_Inhand_Salary': ["https://www.cnbc.com/select/how-does-salary-and-income-impact-your-credit-score/"],
                'Num_Bank_Accounts': ["https://www.mybanktracker.com/credit-cards/credit-score/bank-accounts-hurt-credit-score-20174#:~:text=Quick%20answer%3A%20Credit%20scores%20are%20not%20affected%20by,are%20based%20on%20data%20on%20your%20credit%20report."],
                'Num_Credit_Card': ["https://www.investopedia.com/ask/answers/07/credit_score.asp"],
                'Interest_Rate': ["https://www.experian.com/blogs/ask-experian/do-lower-interest-rates-affect-your-credit-score/"],
                'Num_of_Loan': ["https://www.americanexpress.com/en-us/credit-cards/credit-intel/does-applying-for-multiple-loans-affect-your-credit-score/"],
                'Type_of_Loan': ["https://www.experian.com/blogs/ask-experian/credit-education/improving-credit/improve-credit-score/?msockid=1e7a629a1c436b37287c6db51dc76aa2"],
                'Delay_from_due_date': ["https://www.nerdwallet.com/article/finance/late-bill-payment-reported"],
                'Num_of_Delayed_Payment':["https://www.credit.com/blog/5-myths-about-late-payments-your-fico-scores-71720/"],
                'Changed_Credit_Limit': ["https://www.cnbc.com/select/does-requesting-a-credit-limit-increase-affect-your-credit-score/?msockid=1e7a629a1c436b37287c6db51dc76aa2"],
                'Num_Credit_Inquiries': ["https://www.experian.com/blogs/ask-experian/do-multiple-loan-inquiries-affect-your-credit-score/?msockid=1e7a629a1c436b37287c6db51dc76aa2"],
                'Credit_Mix': ["https://www.experian.com/blogs/ask-experian/credit-education/improving-credit/improve-credit-score/?msockid=1e7a629a1c436b37287c6db51dc76aa2"],
                'Outstanding_Debt': ["https://www.thebalancemoney.com/how-your-debt-affects-your-credit-score-960489","https://www.forbes.com/advisor/credit-cards/debt-snowball-vs-debt-avalanche-the-best-way-to-pay-off-credit-card-debt/"],
                'Credit_Utilization_Ratio': ["https://www.businessinsider.com/personal-finance/credit-score/credit-utilization-ratio#:~:text=1%20Your%20credit%20utilization%20ratio%20is%20the%20percentage,utilization%20ratio%20is%20best%20for%20your%20credit%20scores."],
                'Credit_History_Age': ["https://www.experian.com/blogs/ask-experian/credit-education/improving-credit/improve-credit-score/?msockid=1e7a629a1c436b37287c6db51dc76aa2"],
                'Payment_of_Min_Amount': ["https://www.thebalancemoney.com/the-impact-of-minimum-payments-on-your-credit-score-960463"],
                'Total_EMI_per_month': ["https://www.getonecard.app/blog/when-to-use-credit-card-emi-and-when-to-avoid/"],
                'Amount_invested_monthly': ["https://www.chase.com/personal/credit-cards/education/credit-score/do-investments-affect-your-credit-score"],
                'Payment_Behaviour': ["https://www.capitalone.com/learn-grow/money-management/payment-history/"],
                'Monthly_Balance': ["https://www.experian.com/blogs/ask-experian/better-pay-off-credit-card-full-every-month-or-maintain-balance/?msockid=1e7a629a1c436b37287c6db51dc76aa2"]}
    for feature, advice in advice_dict.items():
        resource_links = advice_resources.get(feature, ["No specific resources available."])
        resource_text = "\n".join(resource_links)

        if not advice or advice == []:
            advice_dict[feature] =  [resource_text]
        else:
            advice_dict[feature] = [advice[0], resource_text]


    return advice_dict


def preprocess(): 
    '''
    Function to fetch raw data and output the preprocessed cleaned data. 
    '''
    df = pd.read_csv("data/train.csv")
    label_encoder = LabelEncoder()

    # Preprocessing Data
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


    # Remove the top 1% of rows with the highest annual income.
    quantiles = df['Annual_Income'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.998])
    q99 = quantiles[0.99]
    df_filtered = df[df['Annual_Income'] <= q99].copy()
    # Remove the top 2% of rows with the most bank accounts.
    quantiles = df_filtered['Num_Bank_Accounts'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995])
    q98 = quantiles[0.98]
    df_filtered = df_filtered[(df_filtered['Num_Bank_Accounts'] <= q98) & (df_filtered["Num_Bank_Accounts"] >= 0)]
    # Remove the top 3% of rows with the highest number of credit cards.
    quantiles = df_filtered['Num_Credit_Card'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995])
    q97 = quantiles[0.97]
    df_filtered = df_filtered[(df_filtered['Num_Credit_Card'] <= q97) & (df_filtered["Num_Credit_Card"] >= 0)]
    # Remove the top 2% of rows with the most credit inquiries.
    quantiles = df_filtered['Num_Credit_Inquiries'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995])
    q98 = quantiles[0.98]
    df_filtered = df_filtered[(df_filtered['Num_Credit_Inquiries'] <= q98) & (df_filtered["Num_Credit_Inquiries"] >= 0)]
    # Remove the top 0.47% of rows with the most loans.
    quantiles = df_filtered['Num_of_Loan'].quantile([0.25, 0.50, 0.75, 0.90, 0.97, 0.98, 0.99, 0.995, 0.9951, 0.9953, 0.9954, 0.996, 0.998])
    q9953 = quantiles[0.9953]
    df_filtered = df_filtered[(df_filtered['Num_of_Loan'] <= q9953) & (df_filtered["Num_of_Loan"] >= 0)]
    #Remove the top 0.8% of rows with the highest number of delayed payments.
    quantiles = df_filtered['Num_of_Delayed_Payment'].quantile([0.25, 0.50, 0.75, 0.90, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.995])
    q992 = quantiles[0.992]
    df_filtered = df_filtered[(df_filtered['Num_of_Delayed_Payment'] <= q992) & (df_filtered["Num_of_Delayed_Payment"] >= 0)]
    # Filter out rows with negative monthly balance values.
    quantiles = df_filtered['Monthly_Balance'].quantile([0.25, 0.50, 0.75, 0.90, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.995, 0.998, 0.9999])
    df_filtered = df_filtered[(df_filtered["Monthly_Balance"] >= 0)]
    quantiles = df_filtered['Amount_invested_monthly'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.952, 0.954, 0.956, 0.958, 0.96])
    q954 = quantiles[0.954]
    df_filtered = df_filtered[(df_filtered['Amount_invested_monthly'] <= q954) & (df_filtered["Amount_invested_monthly"] >= 0)]
    quantiles = df_filtered['Total_EMI_per_month'].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.96, 0.965, 0.968, 0.99, 0.995])
    q965 = quantiles[0.965]
    df_filtered = df_filtered[(df_filtered['Total_EMI_per_month'] <= q965) & (df_filtered["Total_EMI_per_month"] >= 0)]
    # Handle abnormal values in age by setting out-of-range ages to NaN and interpolating.
    df_filtered.loc[(df_filtered['Age'] < 0) | (df_filtered['Age'] > 100), 'Age'] = pd.NA
    df_filtered['Age'] = df_filtered.groupby('Name')['Age'].transform(lambda x: x.interpolate(method='linear'))
    # Fill missing values for Monthly Inhand Salary using linear interpolation.
    df_filtered['Monthly_Inhand_Salary'] = df_filtered.groupby('Name')['Monthly_Inhand_Salary'].transform(lambda x: x.interpolate(method='linear'))
    # Drop rows with missing values in Age, Outstanding Debt, and Monthly Balance.
    df_filtered.dropna(subset=["Age", "Outstanding_Debt", "Monthly_Balance"], inplace=True)

    #Calculating average values
    avg_age_per_person = df_filtered.groupby('Name')['Age'].mean()
    avg_aincome_pperson = df_filtered.groupby('Name')['Annual_Income'].mean()
    avg_monthly_inhand_salary_pperson = df_filtered.groupby('Name')['Monthly_Inhand_Salary'].mean()
    avg_num_bank_account_pperson = df_filtered.groupby('Name')['Num_Bank_Accounts'].mean()
    avg_num_credit_card_pperson = df_filtered.groupby('Name')['Num_Credit_Card'].mean()
    avg_num_loan_pperson = df_filtered.groupby('Name')['Num_of_Loan'].mean()
    avg_delay_from_due_date_pperson = df_filtered.groupby('Name')['Delay_from_due_date'].mean()
    avg_num_delayed_payment_pperson = df_filtered.groupby('Name')['Num_of_Delayed_Payment'].mean()
    avg_num_credit_inquiries_pperson = df_filtered.groupby('Name')['Num_Credit_Inquiries'].mean()
    avg_outstanding_debt_pperson = df_filtered.groupby('Name')['Outstanding_Debt'].mean()
    avg_credit_util_ratio_pperson = df_filtered.groupby('Name')['Credit_Utilization_Ratio'].mean()
    avg_credit_history_age_pperson = df_filtered.groupby('Name')['Credit_History_Age_in_Years'].mean()
    avg_credit_history_age_pperson = df_filtered.groupby('Name')['Payment_of_Min_Amount'].mean()
    avg_monthly_investment_pperson = df_filtered.groupby('Name')['Amount_invested_monthly'].mean()
    avg_monthly_balance_pperson = df_filtered.groupby('Name')['Monthly_Balance'].mean()
    credit_score_pperson = df_filtered['Credit_Score']

    #Calculating correlations
    df_numeric = df_filtered.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = df_numeric.corr()
    cross_tab = pd.crosstab(credit_score_pperson, df_filtered["Payment_of_Min_Amount"])

    back_df = df_filtered.copy()
    df_filtered = back_df.copy()
    dict = {
        'High_spent_Small_value_payments' : 0,
        'Low_spent_Large_value_payments' : 1,
        'Low_spent_Medium_value_payments' : 2,
        'Low_spent_Small_value_payments' : 3,
        'High_spent_Medium_value_payments' : 4,
        'High_spent_Large_value_payments': 5,
        '!@9#%8' : np.nan
    }

    df_filtered['Payment_Behaviour'] = df_filtered['Payment_Behaviour'].map(dict)
    label_encoder = LabelEncoder()

    df_filtered.drop("ID", axis=1, inplace=True)
    df_filtered.drop("Name", axis=1, inplace=True)
    df_filtered.drop("Customer_ID", axis=1, inplace=True)
    df_filtered.drop("SSN", axis=1, inplace=True)
    df_filtered.drop("Type_of_Loan", axis=1, inplace=True)
    df_filtered.drop("Monthly_Inhand_Salary", axis=1, inplace=True)
    df_filtered.drop("Credit_History_Age", axis=1, inplace=True)
    df_filtered.drop("Month", axis=1, inplace=True)
    df_filtered.drop(['Monthly_Balance', 'Credit_Utilization_Ratio', "Interest_Rate", "Occupation", "Age"], axis=1, inplace=True)
    print(df_filtered.isna().sum())

    df_filtered = df_filtered.dropna()
    return [df_filtered, correlation_matrix]