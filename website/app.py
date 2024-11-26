from flask import Flask, render_template, request
import time

app = Flask(__name__)

# Simulate a time-consuming function
def time_consuming_function(inputs):
    time.sleep(5)  # Simulate processing time
    credit_level = "Awesome"
    list_of_suggestions = ["The sky is blue", "The Ocean is deep", "The world is beautiful"]
    some_useful_links = ["https://google.com", "https://openai.com", "https://anthropic.com"]
    processed_result = list(zip(list_of_suggestions, some_useful_links))
    return [credit_level, processed_result]

@app.route('/', methods=['GET', 'POST'])
def index():
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
            'Credit_History_Age': request.form.get('Credit_History_Age'),
            'Payment_of_Min_Amount': request.form.get('Payment_of_Min_Amount'),
            'Total_EMI_per_month': request.form.get('Total_EMI_per_month'),
            'Amount_invested_monthly': request.form.get('Amount_invested_monthly'),
            'Payment_Behaviour': request.form.get('Payment_Behaviour'),
            'Monthly_Balance': request.form.get('Monthly_Balance'),
        }

        # Process the inputs
        results = time_consuming_function(inputs)

        # Render the result on the same page
        return render_template('index.html', credit_level = results[0], result=results[1])

    # Render the initial form page
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
