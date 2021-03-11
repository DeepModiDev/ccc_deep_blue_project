from django.contrib.auth.admin import User
from django.contrib.auth.forms import (UserCreationForm,UserChangeForm)
from django import forms

class CustomUserCreationForm(UserCreationForm):
    email = forms.CharField(required=True, widget=forms.EmailInput(attrs={'class': 'validate', }))
    class Meta:
        model = User
        fields=['first_name','last_name','username','email','is_active']

class CustomUserChageForm(UserChangeForm):
    password = None
    email = forms.CharField(required=True, widget=forms.EmailInput(attrs={'class': 'validate', }))
    class Meta:
        model = User
        fields=['first_name','last_name','email','is_active',]
