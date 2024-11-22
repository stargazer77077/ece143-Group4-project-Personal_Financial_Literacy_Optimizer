from django import forms
class InputForm(forms.Form):
    input_text = forms.CharField(label = 'Enter some text')