from django.shortcuts import render
from .forms import InputForm
# Create your views here.

def calculate_data(input_1): 
    return f"{input_1}"

def form_view(request):
    output = None
    if request.method == 'POST': 
        form = InputForm(request.POST)
        if form.is_valid(): 
            input1 = form.cleaned_data['input_text']
            output = calculate_data(input1)
    else: 
        form = InputForm()
    return render(request, 'form.html', {'form': form, 'output': output})