import django_filters
from django import forms
from django.contrib.auth.models import User
from .models import Images,Videos,ImageDetails,DetectionVideos   
from django_filters.widgets import RangeWidget

class ImageFilter(django_filters.FilterSet):
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type': 'date'}))
    class Meta:
        model = Images
        fields = ['date']

class VideoFilter(django_filters.FilterSet):
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type':'date'}))
    class Meta:
        model = DetectionVideos
        fields = ['date']