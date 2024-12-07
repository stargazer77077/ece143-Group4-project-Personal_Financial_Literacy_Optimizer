import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from hyperopt import fmin, tpe, hp, Trials
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from helper_functions import *
print("New Code")

def process_data(input_data):
    '''
    Main model processing function which controls model processing flow. 
    '''
    # List of features to use.
    model_feature_list = ['Annual_Income',
                        'Num_Bank_Accounts',
                        'Num_Credit_Card',
                        'Num_of_Loan',
                        'Delay_from_due_date',
                        'Num_of_Delayed_Payment',
                        'Changed_Credit_Limit',
                        'Num_Credit_Inquiries',
                        'Credit_Mix',
                        'Outstanding_Debt',
                        'Payment_of_Min_Amount',
                        'Total_EMI_per_month',
                        'Amount_invested_monthly',
                        'Payment_Behaviour',
                        'Credit_History_Age_in_Years']

    # Get preprocessed data as a data frame and load the pretrained model.
    df_filtered, correlation_matrix = preprocess()
    best_model = joblib.load("ExtraTreeClf_sklearn_latest.pkl")

    # Process the input to make it usable by the model.
    sc = StandardScaler()
    X = df_filtered.drop(['Credit_Score'], axis=1)
    X_scaled = sc.fit_transform(X)
    current_sample = input_data
    proper_sample = dict()
    for feature_to_use in model_feature_list:
        proper_sample[feature_to_use] = float(current_sample[feature_to_use])
    current_sample = pd.Series(proper_sample)

    # Setup optimizer class.  
    optimizer = CreditScoreOptimizer(current_sample=current_sample, X_train=X, sc=sc, best_model=best_model)
    optimized_params, best_score = optimizer.progressive_optimize(correlation_matrix, base_max_evals=10)
    features = ["Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                'Type_of_Loan',  'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix',
                'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age', 'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly',
                'Payment_Behaviour', 'Monthly_Balance']
    # Use optimizer class to generate predictions and suggestions.
    current_credit_score = str(optimizer.get_current_pred(current_sample))
    advice_dict = initialize_advice_dict(features)
    original_sample = current_sample.to_dict()
    optimized_advice = optimizer.generate_advice(optimized_params, original_sample)
    for feature, advice in optimized_advice.items():
        advice_dict[feature].append(advice)

    #Final Advice Dictionary Output
    advice_dict = append_resources(advice_dict)
    return [current_credit_score, advice_dict]