from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import os
from .imagePrediction import ImagePrediction
from .videoPrediction import VideoPrediction
from .models import Images, Videos, ImageDetails
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from datetime import datetime


@login_required
def home(request):
    context = {}
    count = User.objects.count()
    if request.user.is_superuser:
        users = User.objects.all()
        context['users'] = users
        context['isSuperUser'] = True
        context['usersCount'] = count
    else:
        context['count'] = count
        context['isVisible'] = False
        context['usersCount'] = count
    return render(request, 'home.html',context)


@login_required
def upload(request):
    urlsList = []
    predictedDataDetails = []
    isValidMessage = []

    if request.method == 'POST':
        uploaded_file = request.FILES.getlist("document")
        for image in uploaded_file:

            (isValid, type, size) = sizeValidator(image.size)

            if isValid:
                if validator(image.name):

                    # print("Extension match: ", validator(image.name))
                    # fs = FileSystemStorage()
                    # name = fs.save(image.name, image)
                    # print("image name: ",image.name)
                    # print("image path:",os.path.join(os.getcwd(), 'media/images/', image.name))
                    # newName = str(currentTime.day)+"_"+str(currentTime.month)+"_"+str(currentTime.year)+"_"+str(currentTime.second)+"_"+image.name
                    # currentTime = datetime.now()

                    currentUser = request.user
                    savingImage = Images(user=currentUser, image=image)
                    savingImage.save()

                    newName = savingImage.image.name
                    imagePrediction = ImagePrediction()

                    imagePrediction.get_predection(os.path.join(os.getcwd(), 'media/', newName))
                    predictedDataDetails.append("Mannequin found: " + str(
                        imagePrediction.getPredictedMannequinCount()) + ",  Person found: " + str(
                        imagePrediction.getPredictedPersonCount()))
                    urlsList.append(newName)

                    imageDetails = ImageDetails(image=savingImage,imageDetails="Mannequin found: " + str(
                        imagePrediction.getPredictedMannequinCount()) + ",  Person found: " + str(
                        imagePrediction.getPredictedPersonCount()))
                    imageDetails.save()

                else:
                    isValidMessage.append({'message': 'Invalid file format. Please select images only.',
                                           'items': "Problem with:" + image.name})

            else:
                isValidMessage.append({'message': 'Image size must be less than ' + str(size) + " " + type,
                                       'items': "Problem with:" + image.name})

        context = {
            "url": [{"link": "/media/" + item, "itemId": index} for index, item in enumerate(urlsList)],
            "predictedData": [{"outputData": item, "itemId": index} for index, item in enumerate(predictedDataDetails)],
            "errorMessage": isValidMessage
        }
        return render(request, 'upload.html', context)

    if request.method == 'GET':
        currentUser = request.user
        print("Current userid in get method of image: ", currentUser.id)
        return render(request, 'upload.html', context={})


@login_required
def video(request):
    context = {}
    isValidMessage = []

    if request.method == 'POST':
        uploaded_file = request.FILES['document']

        if validatorVideo(uploaded_file.name):

            (isValid, type, size) = sizeValidatorVideo(uploaded_file.size)

            if isValid:
                title = uploaded_file.name
                # fs = FileSystemStorage()
                # fs.save('videos\\'+uploaded_file.name, uploaded_file)
                currentUser = request.user
                video = Videos(user_id=currentUser.pk, video=uploaded_file)
                video.save()

                videoPrediction = VideoPrediction()
                videoPrediction.setVideoTitle(video.video.name.split('/')[1])
                videoPrediction.setvideoURL(os.path.join(os.getcwd(), 'media/', video.video.name))
                videoPrediction.setuserId(currentUser.pk)
                videoPrediction.caller()

                video.delete()
                context['processedVideoUrl'] = videoPrediction.getDetectedVideoUrl()
                print(context)
                return render(request, 'video.html', context)
            else:
                isValidMessage.append({'message': 'Video size must be less than ' + str(size) + " " + type,
                                       'items': "Problem with:" + uploaded_file.name})
        else:
            isValidMessage.append({'message': 'Invalid file format. Please select video only.',
                                   'items': "Problem with:" + uploaded_file.name})

        context['errorMessage'] = isValidMessage
        return render(request, 'video.html', context)

    if request.method == 'GET':
        return render(request, 'video.html')


@login_required
def feedURL(request):
    context = {}
    if request.method == 'POST':
        feededURL = request.POST['feedURL']
        feedPrediction = VideoPrediction()
        feedPrediction.setfeedURL(feededURL)
        feedPrediction.feedVideo()
    return render(request, 'video.html', context)


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})


def validator(fileName):
    extensions = [".jpeg", ".jpg", ".png", ".PNG", ".JPG", ".JPEG", ".webp", ".WEBP"]
    (_, fileExtension) = os.path.splitext(fileName)
    if fileExtension in extensions:
        return True
    else:
        return False


def validatorVideo(fileName):
    extensions = [".mp4", ".avi"]
    (_, fileExtension) = os.path.splitext(fileName)
    if fileExtension in extensions:
        return True
    else:
        return False


def sizeValidator(fileSize):
    size, type = convert_bytes(fileSize)
    tags = {"Byte": 1e+7, "Kilobyte": 10000, "Megabyte": 10, "Gigabyte": 0.01, "Terabyte": 1e-5}
    if tags[type] >= size:
        return (True, type, tags[type])
    else:
        return (False, type, tags[type])


def sizeValidatorVideo(fileSize):
    size, type = convert_bytes(fileSize)
    tags = {"Byte": 1e+7, "Kilobyte": 10000, "Megabyte": 100, "Gigabyte": 0.01, "Terabyte": 1e-5}
    if tags[type] >= size:
        return (True, type, tags[type])
    else:
        return (False, type, tags[type])


def convert_bytes(bytes_number):
    tags = ["Byte", "Kilobyte", "Megabyte", "Gigabyte", "Terabyte"]
    i = 0
    double_bytes = bytes_number

    while (i < len(tags) and bytes_number >= 1024):
        double_bytes = bytes_number / 1024.0
        i = i + 1
        bytes_number = bytes_number / 1024

    return (round(double_bytes, 2), tags[i])


def load_videos():
    videos = Videos.objects.all()
    context = {
        'videos': videos,
    }
    return context

# if request.method == 'POST':
#     uploaded_file = request.FILES['document']
#     title = uploaded_file.name
#     video = Videos(title=title,video=uploaded_file)
#     video.save()
#     videoPath = video.video.url
#     videoPrediction = VideoPrediction()
#     videoPrediction.setvideoURL(os.path.join(os.getcwd(),'media',video.title))
#     videoPrediction.setVideoTitle(title)
#     videoPrediction.ControlledThreading()
#     context['url'] = videoPath
#     #
#     print(context)
# return render(request, 'video.html', context)

# if request.method == 'POST':
#     uploaded_file = request.FILES['document']
#     #fs = FileSystemStorage()
#     #fs.save('videos\\'+uploaded_file.name, uploaded_file)
#     title = uploaded_file.name
#     video = Videos(title=title,video=uploaded_file)
#     video.save()
#     videoPrediction = VideoPrediction()
#     videoPrediction.setVideoTitle(uploaded_file.name)
#     videoPrediction.setvideoURL(os.path.join(os.getcwd(), 'media\\videos',uploaded_file.name))
#     videoPrediction.ControlledThreading()
#     print("context",context)
#     context['url'] = os.path.join(os.getcwd(), 'media\\videos',uploaded_file.name)
# return render(request, 'video.html', context)
