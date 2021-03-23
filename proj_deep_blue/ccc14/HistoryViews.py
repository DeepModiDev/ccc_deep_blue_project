from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from .models import DetectionVideos, Images, ImageDetails,MallOwnerShopOwner
from django.http import JsonResponse
from .filters import ImageFilter,VideoFilter,ImageFilterForAdmin,VideoFilterForAdmin
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
            videoList.append({'videoURL': video.video.url, 'videoTitle': video.videoTitle, 'videoId': video.pk, 'videoThumbnail': video.thumbnail.url,'date':video.date,'minPersonCount':video.min_count,'maxPersonCount':video.max_count,'avgPersonCount':video.average_count,'medianPersonCount':video.median_count})

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

        image_filter = ImageFilter(request.GET,queryset=usersImages,request=request)
        usersImages = image_filter.qs # here qs means query set

        for images in usersImages:
            imageDetails = ImageDetails.objects.get(image_id=images.pk)
            imageList.append({'date':images.date,'imageURL':images.image.url,'imageTitle':images.imageTitle,'imageId':images.pk,'imageDetails':imageDetails.imageDetails})

        context = {
            "images":imageList,
            "image_filter":image_filter,
        }
        return render(request,'history/images.html',context)

@login_required(login_url="/accounts/login/")
@staff_member_required
def users_uploaded_videos(request):
    context = {}
    users = []
    videoFiles = []
    videoDetailsList = []

    if request.method == "GET":
        if request.user.is_staff:
            context['isMallOwner'] = True

            mallOwner = User.objects.get(id=request.user.pk)
            mallOwnerShopOwners = MallOwnerShopOwner.objects.all()

            for owners in mallOwnerShopOwners:
                if (owners.mallOwner.pk == request.user.pk):
                    user = User.objects.get(pk=owners.shopOwner.pk)
                    users.append(user)
            users.append(mallOwner)

            allVideos = DetectionVideos.objects.all()
            for user in users:
                for video in allVideos:
                    if user.pk == video.user_id:
                        videoFiles.append(video.pk)

            videoFiles = DetectionVideos.objects.filter(pk__in=videoFiles)
            videoFilterForAdmin = VideoFilterForAdmin(request.GET,request=request,queryset=videoFiles)
            videoFiles = videoFilterForAdmin.qs

            for videos in videoFiles:
                user = User.objects.get(id=videos.user.id)
                videoDetailsList.append(
                    {'videoURL': videos.video.url, 'videoTitle': videos.videoTitle, 'videoId': videos.pk, 'videoThumbnail': videos.thumbnail.url,'date':videos.date,'user':user})

            context['VideoFiles'] = videoDetailsList
            context['video_filter_admin'] = videoFilterForAdmin

            return render(request,"history/UsersVideos.html",context)

@login_required(login_url="/accounts/login/")
@staff_member_required
def users_uploaded_images(request):
    context = {}
    imageDetailList = []
    users = []
    imageFiles=[]
    if request.method == "GET":
        if request.user.is_staff:
            context['isMallOwner'] = True

            mallOwner = User.objects.get(id=request.user.pk)
            mallOwnerShopOwners = MallOwnerShopOwner.objects.all()
            for owners in mallOwnerShopOwners:
                if (owners.mallOwner.pk == request.user.pk):
                    user = User.objects.get(pk=owners.shopOwner.pk)
                    users.append(user)
            users.append(mallOwner)

            allImages = Images.objects.all()

            for user in users:
                for image in allImages:
                    if user.pk == image.user_id:
                        imageFiles.append(image.pk)

            imageFiles = Images.objects.filter(pk__in=imageFiles)
            imageFilterForAdmin = ImageFilterForAdmin(request.GET,request=request,queryset=imageFiles)
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
        data = {}
        data['deleted'] = True
        return JsonResponse(data)

@login_required(login_url='/accounts/login/')
def delete_image_by_admin(request,pk):
    if request.method == "POST":
        image = Images.objects.get(pk=pk)
        image.delete()
        data = {}
        data['deleted'] = True
        return JsonResponse(data)

@login_required(login_url='/accounts/login/')
def delete_video_by_admin(request,pk):
    if request.method == "POST":
        video = DetectionVideos.objects.get(pk=pk)
        video.delete()
        data = {}
        data['deleted'] = True
        return JsonResponse(data)