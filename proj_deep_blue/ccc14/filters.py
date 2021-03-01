import django_filters
from django import forms
from django.contrib.auth.models import User
from .models import Images,Videos,ImageDetails,DetectionVideos   
from django_filters.widgets import RangeWidget

class ImageFilter(django_filters.FilterSet):
    imageTitle = django_filters.CharFilter(label="Search by name",
                                           widget=forms.TextInput(attrs={'type': 'text', 'class': 'small'}))
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type': 'date'}))
    class Meta:
        model = Images
        fields = ['imageTitle','date']

class VideoFilter(django_filters.FilterSet):
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type':'date'}))
    class Meta:
        model = DetectionVideos
        fields = ['date']


def get_valve_choices():
    return [[mUser.pk, mUser] for mUser in User.objects.all()]
class ImageFilterForAdmin(django_filters.FilterSet):

    user = django_filters.ModelMultipleChoiceFilter(queryset=User.objects.all(),widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-control'}),label="Users")
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type': 'date','class':'small'}))
    imageTitle = django_filters.CharFilter(label="Search by name",widget=forms.TextInput(attrs={'type':'text','class':'small'}))

    class Meta:
        model = Images
        fields = ['user','date','imageTitle']