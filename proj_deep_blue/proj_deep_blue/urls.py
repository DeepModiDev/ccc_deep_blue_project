from django.contrib import admin
from django.urls import path, include,re_path
from django.conf import settings
from django.conf.urls.static import static
from ccc14 import views,HistoryViews
from django.views.static import serve
from django.conf.urls import handler500,handler404,handler403,handler400

urlpatterns = [
    #path('', views.Home.as_view(), name='home'),
    path('', views.signup, name='signup'),  # for sign up
    path('home/', views.home, name='home'), # for home page
    path('upload/', views.upload, name='upload'), #for uploading the images
    path('video/',views.video,name='video'),    #for uploading the videos
    path('feed/',views.feedURL,name='feedURL'), #custom feed
    path('admin/', admin.site.urls),    #Admin Url
    path('accounts/', include('django.contrib.auth.urls')), # for login
    path('history/videos/',HistoryViews.videos,name='historyVideos'),    #History Images
    path('history/images/',HistoryViews.images,name='historyImages'),    #History Video
    path('history/images/<int:pk>',HistoryViews.delete_image,name="deleteImage"), #Delete Image
    path('history/videos/<int:pk>',HistoryViews.delete_video,name="deleteVideo"), #Delete Image
    re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
]

urlpatterns  += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

handler404 = 'ccc14.ErrorHandlingView.error_404'
handler500 = 'ccc14.ErrorHandlingView.error_500'
handler400 = 'ccc14.ErrorHandlingView.error_400'
handler403 = 'ccc14.ErrorHandlingView.error_403'