import kagglehub
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np

def data_process(data_path=None):
    if data_path is None:
      print("=" * 20 + "Downloading dataset..." + "=" * 20)
      path = kagglehub.dataset_download("parisrohan/credit-score-classification")
      print("Path to dataset files:", path)

    label_encoder = LabelEncoder()

    print("=" * 20 + "Loading data..." + "=" * 20)
    if data_path is None:
      df = pd.read_csv(f"{path}\\train.csv")
    else:
      df = pd.read_csv(data_path)

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
    return df_filtered

def initialize_advice_dict(features):
    """
    Initializes an advice dictionary with each feature as a key and an empty list.

    Parameters:
      features (list): List of feature names.

    Returns:
      dict: Dictionary with feature keys and empty lists as values.
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
            # If advice is empty, only add the resource links
            advice_dict[feature] = f"Here are some resources on how {feature} might affect your credit score:\n{resource_text} \n"
        else:
            # Append resource links to the existing advice for this feature
            advice_dict[feature] = f"{advice[0]}\nHere are some resources on how {feature} might affect your credit score and ways to adjust them:\n{resource_text}\n"


    return advice_dict