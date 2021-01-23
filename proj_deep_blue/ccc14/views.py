from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
import os
from .imagePrediction import ImagePrediction
from .videoPrediction import VideoPrediction

class Home(TemplateView):
    template_name = 'home.html'

def upload(request):
    urlsList = []
    predictedDataDetails = []
    imagePrediction = ImagePrediction()
    if request.method == 'POST':
        uploaded_file = request.FILES.getlist("document")
        for image in uploaded_file:
            fs = FileSystemStorage()
            name = fs.save(image.name, image)
            imagePrediction.get_predection(os.path.join(os.getcwd(), 'media', name))
            predictedDataDetails.append("Mannequin found: "+str(imagePrediction.getPredictedMannequinCount()) + ",  Person found: "+str(imagePrediction.getPredictedPersonCount()))
            urlsList.append(name)
    context = {
        "url":[{"link":"/media/"+item,"itemId":index} for index,item in enumerate(urlsList)],
        "predictedData":[{"outputData":item,"itemId":index} for index,item in enumerate(predictedDataDetails)],
    }
    return render(request, 'upload.html', context)

def video(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        videoPrediction = VideoPrediction()
        videoPrediction.setvideoURL(os.path.join(os.getcwd(), 'media', name))
        videoPrediction.caller()
        context['url'] = fs.url(name)
    return render(request, 'video.html', context)

def feedURL(request):
    context = {}
    if request.method == 'POST':
        feededURL = request.POST['feedURL']
        feedPrediction = VideoPrediction()
        feedPrediction.setfeedURL(feededURL)
        feedPrediction.feedVideo()
        print(feededURL)
    return render(request,'video.html',context)
