from flask import Flask, render_template, request
import time
from model_code import *


app = Flask(__name__)

def push_advice_to_website(inputs):
    '''
    Function to enter the user input to the model and format the model output to display on website. 
    '''
    model_outputs = process_data(inputs)
    credit_level = model_outputs[0]
    suggestion_data = model_outputs[1]
    list_of_concrete_suggestions = []
    list_of_link_suggestions = []
    list_of_links = []
    for key1 in suggestion_data: 
        if len(suggestion_data[key1]) == 2: 
            string1 = suggestion_data[key1][0]
            string1 = string1.replace("_", " ")
            list_of_concrete_suggestions = list_of_concrete_suggestions + [string1]
            list_of_links = list_of_links + [suggestion_data[key1][1]]
        elif len(suggestion_data[key1]) == 1: 
            string1 = f"You can find more data about {key1} here:"
            string1 = string1.replace("_", " ")
            list_of_link_suggestions = list_of_link_suggestions + [string1]
            list_of_links = list_of_links + [suggestion_data[key1][0]]
    list_of_suggestions = list_of_concrete_suggestions + list_of_link_suggestions
    some_useful_links = list_of_links
    processed_result = list(zip(list_of_suggestions, some_useful_links))
    return [credit_level, processed_result]

@app.route('/', methods=['GET', 'POST'])
def index():
    '''
    Function to control the user interface of the website.
    '''
    if request.method == 'POST':
        # Collect all form inputs into a dictionary
        inputs = {
            'Age': request.form.get('Age'),
            'Occupation': request.form.get('Occupation'),
            'Annual_Income': request.form.get('Annual_Income'),
            'Monthly_Inhand_Salary': request.form.get('Monthly_Inhand_Salary'),
            'Num_Bank_Accounts': request.form.get('Num_Bank_Accounts'),
            'Num_Credit_Card': request.form.get('Num_Credit_Card'),
            'Interest_Rate': request.form.get('Interest_Rate'),
            'Num_of_Loan': request.form.get('Num_of_Loan'),
            'Type_of_Loan': request.form.get('Type_of_Loan'),
            'Delay_from_due_date': request.form.get('Delay_from_due_date'),
            'Num_of_Delayed_Payment': request.form.get('Num_of_Delayed_Payment'),
            'Changed_Credit_Limit': request.form.get('Changed_Credit_Limit'),
            'Num_Credit_Inquiries': request.form.get('Num_Credit_Inquiries'),
            'Credit_Mix': request.form.get('Credit_Mix'),
            'Outstanding_Debt': request.form.get('Outstanding_Debt'),
            'Credit_Utilization_Ratio': request.form.get('Credit_Utilization_Ratio'),
            'Credit_History_Age_in_Years': request.form.get('Credit_History_Age'),
            'Payment_of_Min_Amount': request.form.get('Payment_of_Min_Amount'),
            'Total_EMI_per_month': request.form.get('Total_EMI_per_month'),
            'Amount_invested_monthly': request.form.get('Amount_invested_monthly'),
            'Monthly_Balance': request.form.get('Monthly_Balance'),
        }
        # Process one of the fields and make it consistent with the model input format. 
        spent_amount = request.form.get("Spent_Amount")
        payment_amount = request.form.get("Payment_Amount")
        print(spent_amount)
        print(payment_amount)
        payment_of_min_amount = spent_amount + "_spent_" + payment_amount  + "_value_payments"
        dict_map_payment_of_min_amount = {
            'High_spent_Small_value_payments' : 0,
            'Low_spent_Large_value_payments' : 1,
            'Low_spent_Medium_value_payments' : 2,
            'Low_spent_Small_value_payments' : 3,
            'High_spent_Medium_value_payments' : 4,
            'High_spent_Large_value_payments': 5}
        final_payment_of_min_amount_conv = dict_map_payment_of_min_amount[payment_of_min_amount]
        inputs['Payment_Behaviour'] = final_payment_of_min_amount_conv
        # Generate data to put on the results page.
        results = push_advice_to_website(inputs)

        # Render the result on the same page
        return render_template('index.html', credit_level = results[0], result=results[1])

    # Render the initial form page
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
