from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from .models import DetectionVideos, Images, ImageDetails
from django.contrib.auth.models import User
from django.http import JsonResponse

@login_required(login_url='/accounts/login/')
def videos(request):
    videoList = []
    if request.method == "GET":
        currentUser = request.user
        load_videos(currentUser.id,videoList)
        context = {
            "videos":videoList
        }
        print("Videos: ",videoList)
        return render(request,'history/videos.html',context)

@login_required(login_url='/accounts/login/')
def images(request):
    imageList = []
    if request.method == "GET":
        currentUser = request.user
        load_images(currentUser.id,imageList)
        context = {
            "images":imageList
        }
        return render(request,'history/images.html',context)


def load_videos(userID,videoList):
    userVideos = DetectionVideos.objects.filter(user_id=userID).order_by("-timestamp")
    for video in userVideos:
        videoList.append({'videoURL':video.video.url,'videoTitle':video.video.name.split('/')[2],'videoId':video.pk,'videoThumbnail':video.thumbnail.url})

def load_images(userID,imageList):
    usersImages = Images.objects.filter(user_id=userID)
    for images in usersImages:
        imageDetails = ImageDetails.objects.get(image_id=images.pk)
        imageList.append({'imageURL':images.image.url,'imageTitle':images.image.name.split('/')[1],'imageId':images.pk,'imageDetails':imageDetails.imageDetails})

@login_required(login_url='/accounts/login/')
def delete_image(request,pk):
    if request.method == "POST":
        image = Images.objects.get(pk=pk)
        image.delete()
    return redirect('historyImages')

@login_required(login_url='/accounts/login/')
def delete_video(request,pk):
    if request.method == "POST":
        video = DetectionVideos.objects.get(pk=pk)
        video.delete()
    return redirect('historyVideos')

@login_required(login_url='/accounts/login/')
def delete_image_by_admin(request,pk):
    if request.method == "POST":
        image = Images.objects.get(pk=pk)
        image.delete()
        data = {}
        data['deleted'] = True
    return JsonResponse(data)