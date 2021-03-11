import django_filters
from django import forms
from django.contrib.auth.models import User
from .models import Images,Videos,ImageDetails,DetectionVideos,MallOwnerShopOwner
from django_filters.widgets import RangeWidget

userList = []

class ImageFilter(django_filters.FilterSet):
    imageTitle = django_filters.CharFilter(label="Search by name",
                                           widget=forms.TextInput(attrs={'type': 'text', 'class': 'small'}))
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type': 'date'}))
    class Meta:
        model = Images
        fields = ['imageTitle','date']

class VideoFilter(django_filters.FilterSet):
    videoTitle = django_filters.CharFilter(label="Search by name",
                                           widget=forms.TextInput(attrs={'type': 'text', 'class': 'small'}))
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type':'date'}))
    class Meta:
        model = DetectionVideos
        fields = ['videoTitle','date']


def get_users_list(request):
    if request is None:
        return User.objects.none()
    mallOwnerPk = request.user.pk
    shopOwners = []
    shopOwners.append(mallOwnerPk)
    for user in MallOwnerShopOwner.objects.filter(mallOwner_id=mallOwnerPk):
        shopOwners.append(user.shopOwner.pk)
    return User.objects.filter(pk__in=shopOwners)

class ImageFilterForAdmin(django_filters.FilterSet):

    user = django_filters.ModelMultipleChoiceFilter(queryset=get_users_list,widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-control'}),label="Users")
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type': 'date','class':'small'}))
    imageTitle = django_filters.CharFilter(label="Search by name",widget=forms.TextInput(attrs={'type':'text','class':'small'}))

    class Meta:
        model = Images
        fields = ['user','date','imageTitle']


class VideoFilterForAdmin(django_filters.FilterSet):

    user = django_filters.ModelMultipleChoiceFilter(queryset=get_users_list,widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-control'}),label="Users")
    date = django_filters.DateFromToRangeFilter(widget=RangeWidget(attrs={'type': 'date','class':'small'}))
    videoTitle = django_filters.CharFilter(label="Search by name",widget=forms.TextInput(attrs={'type':'text','class':'small'}))

    class Meta:
        model = DetectionVideos
        fields = ['user','date','videoTitle']
