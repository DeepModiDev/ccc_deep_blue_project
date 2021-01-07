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

    if request.method == 'POST':
        uploaded_file = request.FILES.getlist("document")
        i = 0
        for image in uploaded_file:
            fs = FileSystemStorage()
            print("UploadImageDebug: ",uploaded_file)
            name = fs.save(image.name, image)
            imagePrediction = ImagePrediction()
            imagePrediction.caller(os.path.join(os.getcwd(), 'media', name))
            predictedDataDetails.append("Mannequin found: "+str(imagePrediction.getPredictedMannequinCount()) + ",  Person found: "+str(imagePrediction.getPredictedPersonCount()))
            urlsList.append(name)
    context = {
        "url":[{"link":"/media/"+item,"itemId":index} for index,item in enumerate(urlsList)],
        "predictedData":[{"outputData":item,"itemId":index} for index,item in enumerate(predictedDataDetails)],
    }
    print(urlsList)
    print(predictedDataDetails)
    print(context)
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
