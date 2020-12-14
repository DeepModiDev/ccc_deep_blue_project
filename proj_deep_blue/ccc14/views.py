from django.shortcuts import render

from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
import os
from .imagePrediction import ImagePrediction
from .videoPrediction import VideoPrediction
class Home(TemplateView):
    template_name = 'home.html'

def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        ImagePrediction.caller(os.path.join(os.getcwd(), 'media', name))
        context['url'] = fs.url(name)
    return render(request, 'upload.html', context)

def video(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        VideoPrediction.caller(os.path.join(os.getcwd(), 'media', name))
        context['url'] = fs.url(name)
    return render(request, 'video.html', context)
