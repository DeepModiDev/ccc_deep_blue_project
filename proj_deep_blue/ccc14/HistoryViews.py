from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from .models import DetectionVideos, Images, ImageDetails
from django.http import JsonResponse
from .filters import ImageFilter,VideoFilter,ImageFilterForAdmin
from django.contrib.auth.models import User

@login_required(login_url='/accounts/login/')
def videos(request):
    videoList = []
    if request.method == "GET":
        currentUser = request.user
        userVideos = DetectionVideos.objects.filter(user_id=currentUser.id).order_by("-date")

        video_filter = VideoFilter(request.GET,queryset=userVideos)
        userVideos = video_filter.qs

        for video in userVideos:
            videoList.append({'videoURL': video.video.url, 'videoTitle': video.video.name.split('/')[2], 'videoId': video.pk, 'videoThumbnail': video.thumbnail.url,'date':video.date})

        context = {
            "videos":videoList,
            "video_filter":video_filter
        }
        return render(request,'history/videos.html',context)

@login_required(login_url='/accounts/login/')
def images(request):
    imageList = []
    if request.method == "GET":
        currentUser = request.user
        usersImages = Images.objects.filter(user_id=currentUser.id)

        image_filter = ImageFilter(request.GET,queryset=usersImages)
        usersImages = image_filter.qs # here qs means query set

        for images in usersImages:
            imageDetails = ImageDetails.objects.get(image_id=images.pk)
            imageList.append({'date':images.date,'imageURL':images.image.url,'imageTitle':images.image.name.split('/')[1],'imageId':images.pk,'imageDetails':imageDetails.imageDetails})

        context = {
            "images":imageList,
            "image_filter":image_filter,
        }
        return render(request,'history/images.html',context)

@login_required(login_url="/accounts/login/")
@staff_member_required
def users_uploaded_videos(request):
    context = {}
    if request.method == "GET":
        if request.user.is_superuser:
            context['isSuperUser'] = True
            detectionvideos = DetectionVideos.objects.all()
            users = User.objects.all()
            context['users'] = users
            context['DetectionVideos'] = detectionvideos
            return render(request,"history/UsersVideos.html",context)

@login_required(login_url="/accounts/login/")
@staff_member_required
def users_uploaded_images(request):
    context = {}
    imageDetailList = []
    if request.method == "GET":
        if request.user.is_superuser:
            context['isSuperUser'] = True
            imageFiles = Images.objects.all()

            imageFilterForAdmin = ImageFilterForAdmin(request.GET,queryset=imageFiles)
            imageFiles = imageFilterForAdmin.qs

            for images in imageFiles:
                imageDetails = ImageDetails.objects.get(image_id=images.pk)
                user = User.objects.get(id=images.user.id)
                imageDetailList.append(
                    {'date': images.date, 'imageURL': images.image.url, 'imageTitle': images.image.name.split('/')[1],
                     'imageId': images.pk, 'imageDetails':imageDetails.imageDetails,'userName':user.username})

            context['ImageFiles'] = imageDetailList
            context['image_filter_admin'] = imageFilterForAdmin
            return render(request,"history/UsersImages.html",context)


@login_required(login_url='/accounts/login/')
def delete_image(request,pk):
    if request.method == "POST":
        image = Images.objects.get(pk=pk)
        image.delete()
        data = {}
        data['deleted'] = True
    return JsonResponse(data)

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