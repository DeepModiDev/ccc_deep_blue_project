from django.shortcuts import render

def error_404(request,exception):
    context = {}
    return render(request,'error/error_404.html',context=context,status=404)

def error_500(request):
    context = {}
    return render(request,'error/error_500.html',context=context,status=500)

def error_400(request,exception):
    context = {}
    return render(request,'error/error_400.html',context=context,status=400)

def error_403(request,exception):
    context = {}
    return render(request,'error/error_403.html',context=context,status=403)

