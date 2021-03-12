from django.contrib import admin
from django.urls import path, include,re_path
from django.conf import settings
from django.conf.urls.static import static
from ccc14 import views,HistoryViews
from django.views.static import serve
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required

urlpatterns = [
    #path('', views.Home.as_view(), name='home'),
    path('', views.signup, name='signup'),  # for sign up
    path('home/', views.home, name='home'), # for home page
    path('upload/', views.upload, name='upload'), #for uploading the images
    path('video/',views.video,name='video'),#for uploading the videos
    path('video/feed/',views.webcam_feed,name='webcam_feed'),
    path('video/trial/feed/<link>/',views.trial_feed,name='trial_feed'),
    path('person-tracking/',views.person_tracking,name='person-tracking'),
    path('admin/', admin.site.urls),    #Admin Url
    path('accounts/', include('django.contrib.auth.urls')), # for login
    path('home/users/reset-password/',login_required(staff_member_required(auth_views.PasswordResetView.as_view(template_name='registration/password_reset.html'))),name='reset_password'),
    path('home/users/reset-password-sent',login_required(staff_member_required(auth_views.PasswordResetDoneView.as_view(template_name='registration/password_reset_sent.html'))),name='password_reset_done'),
    path('reset/<uidb64>/<token>/',auth_views.PasswordResetConfirmView.as_view(template_name='registration/user_side_reset_form.html'),name='password_reset_confirm'),
    path('home/users/reset-password-complete/',auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_completed.html'),name='password_reset_complete'),
    path('history/videos/',HistoryViews.videos,name='historyVideos'),    #History Images
    path('history/images/',HistoryViews.images,name='historyImages'),    #History Video
    path('history/users-videos/',HistoryViews.users_uploaded_videos,name='historyUsersVideos'),    #History for user's uploaded images
    path('history/users-images/',HistoryViews.users_uploaded_images,name='historyUsersImages'),    #History for user's uploaded Videos
    path('history/images/<int:pk>',HistoryViews.delete_image,name="deleteImage"), #Delete Image
    path('history/videos/<int:pk>',HistoryViews.delete_video,name="deleteVideo"), #Delete Image
    path('home/delete/<int:pk>',views.deleteUser,name="deleteUser"), # Delete user
    path('home/images/delete/<int:pk>',HistoryViews.delete_image_by_admin,name="deleteImageByAdmin"), # Delete Image By Admin
    path('home/videos/delete/<int:pk>',HistoryViews.delete_video_by_admin,name="deleteVideoByAdmin"), # Delete Video By Admin
    path('home/add/user/',views.addUser,name="addUser"), #user profile
    path('home/edit/user/<int:pk>',views.edit_user,name='editUser'), #edit user
    re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
]

urlpatterns  += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

handler404 = 'ccc14.ErrorHandlingView.error_404'
handler500 = 'ccc14.ErrorHandlingView.error_500'
handler400 = 'ccc14.ErrorHandlingView.error_400'
handler403 = 'ccc14.ErrorHandlingView.error_403'