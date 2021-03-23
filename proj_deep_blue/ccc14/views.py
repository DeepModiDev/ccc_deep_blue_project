from django.shortcuts import (render, redirect)
import os
import datetime
from .imagePrediction import ImagePrediction
from .videoPrediction import VideoPrediction
from .PersonTracking import PersonTracking
from .models import (Images, Videos, ImageDetails,MallOwnerShopOwner,DetectionVideos)
from django.contrib.auth.forms import (UserCreationForm)
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from .forms import CustomUserCreationForm,CustomUserChageForm
from django.http import JsonResponse,HttpResponseRedirect
from django.http.response import StreamingHttpResponse
from .Camera import VideoCamera
from urllib.parse import urlparse
import gc
from datetime import timedelta

@login_required(login_url='/accounts/login/')
def home(request):
    context = {}
    users = []

    if request.user.is_staff:
        mallOwnerShopOwners = MallOwnerShopOwner.objects.all()
        for owners in mallOwnerShopOwners:
            if(owners.mallOwner.pk == request.user.pk):
                user = User.objects.get(pk=owners.shopOwner.pk)
                users.append(user)

        context['shopOwners'] = users
        context['mallOwner'] = request.user
        context['isMallOwner'] = True
        context['shopOwnersCount'] = len(users)
        context['form'] = CustomUserCreationForm()
        return render(request, 'home.html', context)
    else:
        context['isMallOwner'] = False
        return render(request, 'upload.html', context)

# Delete user task done...
@login_required(login_url='/accounts/login/')
@staff_member_required
def deleteUser(request,pk):
    if request.method == 'POST':
        context = {}
        user = User.objects.get(pk=pk)
        if not user.is_superuser:
            user.delete()
            context['message'] = "User deleted successfully."
            context['deleted'] = True
        else:
            context['message'] = "You can not delete this user because it is a super user."
            context['deleted'] = False
        return JsonResponse(context)

# Add user task done...
@login_required(login_url='/accounts/login/')
@staff_member_required
def addUser(request):
    context = {}
    errors = []
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            #form = form.save(commit=False)
            #form.parentUser_id = request.user.pk
            savedForm = form.save()
            shopOwner = User.objects.get(id=savedForm.pk)
            mallOwnerShopOwner = MallOwnerShopOwner(mallOwner=request.user,shopOwner=shopOwner)
            mallOwnerShopOwner.save()
            errors.append("User created successfully.")
            context['created'] = True
            context['message'] = errors
            return JsonResponse(context)
        else:
            context['created'] = False
            errors.append("Fail to register user.")
            if len(form.errors) > 0:
               for field in form:
                    for error in field.errors:
                        errors.append(error)
            context['message'] = errors
            return JsonResponse(context)

@login_required(login_url='/account/login/')
@staff_member_required
def edit_user(request,pk):
    context = {}
    if request.method == 'POST':
        user = User.objects.get(id=pk)
        form = CustomUserChageForm(request.POST,instance=user)
        if form.is_valid():
            form.save()
            context['message'] = "Changes saved successfully."

            return HttpResponseRedirect('/home')
        else:
            context['message'] = "Something went wrong!!! try again later."
            return HttpResponseRedirect('/home')
    else:
        context = {}
        user = User.objects.get(id=pk)
        form = CustomUserChageForm(instance=user)
        context['form'] = form
        return render(request,'registration/editUserForm.html',context)

def signup(request):
    if request.user.is_superuser:
        if request.method == "POST":
            form = CustomUserCreationForm(request.POST)
            if form.is_valid():
                form.save()
                return redirect("login")
        else:
            form = UserCreationForm()
            return render(request, 'registration/signup.html', {'form': form})
    else:
        return redirect("login")

@login_required(login_url='/accounts/login/')
def upload(request):
    urlsList = []
    predictedDataDetails = []
    isValidMessage = []
    imageNames = []

    if request.method == 'POST':
        uploaded_file = request.FILES.getlist("document")
        for image in uploaded_file:

            (isValid, type, size) = sizeValidator(image.size)

            if isValid:
                if validator(image.name):
                    currentUser = request.user

                    # get the current time
                    currentTime = datetime.datetime.now()
                    # rename the image before saving
                    image.name = str(currentTime.day)+"_"+str(currentTime.month)+"_"+str(currentTime.year)+"_"+str(currentTime.minute)+"_"+str(currentTime.second)+"_"+str(currentTime.microsecond)+"_"+image.name
                    savingImage = Images(user=currentUser, image=image,date=datetime.datetime.now(),imageTitle=image.name)
                    savingImage.save()

                    newName = savingImage.image.name
                    imagePrediction = ImagePrediction()

                    imagePrediction.get_predection(os.path.join(os.getcwd(), 'media/', newName))

                    predictedDataDetails.append("Mannequin found: " + str(
                        imagePrediction.getPredictedMannequinCount()) + ",  Person found: " + str(
                        imagePrediction.getPredictedPersonCount()))
                    imageNames.append(savingImage.imageTitle)

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
            "data": zip([{"link": "/media/" + item, "itemId": index} for index, item in enumerate(urlsList)],[{"outputData": item, "itemId": index} for index, item in enumerate(predictedDataDetails)],[{"title": item, "itemId": index} for index, item in enumerate(imageNames)]),
            "errorMessage": isValidMessage
        }
        return render(request, 'upload.html', context)

    if request.method == 'GET':
        currentUser = request.user
        return render(request, 'upload.html', context={})

@login_required(login_url='/accounts/login/')
def video(request):
    context = {}
    isValidMessage = []

    if request.method == 'POST':

        uploaded_file = request.FILES['document']

        if validatorVideo(uploaded_file.name):

            (isValid, type, size) = sizeValidatorVideo(uploaded_file.size)

            if isValid:
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
        gc.collect()
        return render(request, 'video.html')

@login_required(login_url='/accounts/login/')
def person_tracking(request):
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

                personTracking = PersonTracking()
                personTracking.setVideoTitle(video.video.name.split('/')[1])
                personTracking.setvideoURL(os.path.join(os.getcwd(), 'media/', video.video.name))
                personTracking.setuserId(currentUser.pk)
                personTracking.caller()

                video.delete()
                context['processedVideoUrl'] = personTracking.getDetectedVideoUrl()
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
        context['footFallCounter'] = True
        return render(request, 'video.html',context)

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
    tags = {"Byte": 1e+7, "Kilobyte": 10000, "Megabyte": 30, "Gigabyte": 0.01, "Terabyte": 1e-5}
    if tags[type] >= size:
        return (True, type, tags[type])
    else:
        return (False, type, tags[type])

def sizeValidatorVideo(fileSize):
    size, type = convert_bytes(fileSize)
    tags = {"Byte": 1e+7, "Kilobyte": 10000, "Megabyte": 500, "Gigabyte": 0.01, "Terabyte": 1e-5}
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

@login_required(login_url='/accounts/login/')
def webcam_feed(request):
    context = {}
    url = None
    if request.method == 'POST':
        url = request.POST.get('feedURL',None)
        stopLiveFeed = request.POST.get('stopFeedBool',False)
        if stopLiveFeed == False:
            url = url.strip()
            urlObj = urlparse(url)
            context['scheme'] = urlObj.scheme
            context['netloc'] = urlObj.netloc
            context['path'] = urlObj.path
            context['parms'] = urlObj.params
            context['query'] = urlObj.query
            context['framgment'] = urlObj.fragment
            context['username'] = urlObj.username
            context['password'] = urlObj.password
            return JsonResponse(context)
        else:
            context['stopFeedBool'] = True
            return JsonResponse(context)
    else:
        return render(request,'LiveVideoFeed.html')

def gen(WebStream):
    while True:
        (livecount,frame) = WebStream.get_frame()
        print("Live count is: ",livecount)
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@login_required(login_url='/accounts/login/')
def trial_feed(request,link):
    if link.find('true') == -1:
        print("is Link null:",link)
        (scheme,netloc,*path) = link.split('_')
        joinedPath = ""
        for _ in path:
            joinedPath += _+'/'
        print("Joined String: ",scheme+'://'+netloc+'/'+joinedPath)

        return StreamingHttpResponse(gen(VideoCamera(scheme + '://' + netloc + '/' + joinedPath,request.user.pk)), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        gc.collect()
        print("gc called..")
        return HttpResponseRedirect('/video/feed/')



'''
Please dont delete these comments they are very important for the future

 below code will help us to convert image frame into base64 string and return response as a JsonResponse
 important line is image_data = base64.b64encode(image).decode('utf-8')
 
            video = cv2.VideoCapture(url)
            if video.isOpened():
                while True:
                    (ret,frame) = video.read()
                    if not ret:
                        break
                    image = cv2.imencode('.jpg', frame)[1].tobytes()
                    image_data = base64.b64encode(image).decode('utf-8')
                    context['images'] = image_data
                video.release()
                cv2.destroyAllWindows()
'''
