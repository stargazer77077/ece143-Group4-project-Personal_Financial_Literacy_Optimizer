from django import forms
from django.forms import TextInput

class InputForm(forms.Form):
    #input_text0 = forms.CharField(label = 'Enter Occupation')
    #input_text1 = forms.CharField(label = 'Enter Annual_Income')
    #input_text2 = forms.CharField(label = 'Enter Occupation')
    #input_text3 = forms.CharField(label = 'Enter Annual_Income')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        field_labels = [
                "Occupation",
                "Annual_Income",
                "Num_Bank_Accounts",
                "Num_Credit_Card",
                "Num_of_Loan",
                "Delay_from_due_date",
                "Num_of_Delayed_Payment",
                "Changed_Credit_Limit",
                "Num_Credit_Inquiries",
                "Credit_Mix",
                "Outstanding_Debt",
                "Payment_of_Min_Amount",
                "Total_EMI_per_month",
                "Amount_invested_monthly",
                "Payment_Behaviour",
                "Credit_Score",
                "Credit_History_Age_in_Years",
            ]
        
        max_length = max(len(label) for label in field_labels)
        field_labels = [label.replace("_", " ").ljust(max_length) for label in field_labels]

            
        # Dynamically add fields to the form
        for label in field_labels:
            field_name = f"{label}"
            #self.fields[field_name] = forms.CharField(label=f"{label}", max_length=100)
            self.fields[field_name] = forms.CharField(widget=forms.TextInput(attrs={'placeholder': f"{label}", 'style': 'width: 300px;', 'class': 'form-control'}))


            



   