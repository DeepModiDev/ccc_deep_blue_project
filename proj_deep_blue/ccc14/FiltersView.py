from django.http import JsonResponse
from django.shortcuts import render, redirect
import os
from django.contrib.auth.models import User
from .models import Images,Videos,ImageDetails,DetectionVideos

@login_required(login_url="accounts/login/")
@staff_member_required
def FilterImagesByUserId(request,pk):
    context = {}
    if request.method == "GET":
        images = Images.objects.filter(pk_in=pk)
        return JsonResponse(context)