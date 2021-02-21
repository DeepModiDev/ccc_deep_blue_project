from django.contrib.auth.admin import User
from django.contrib.auth.forms import UserCreationForm,UserChangeForm

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = User
        fields=['first_name','last_name','username','email','is_active','is_staff','is_superuser']

class CustomUserChageForm(UserChangeForm):
    class Meta:
        model = User
        fields=['first_name','last_name','username','email','is_active','is_staff','is_superuser']