from django.contrib import admin
from .models import Videos,Images,ImageDetails,DetectionVideos

admin.site.register(Videos)
admin.site.register(DetectionVideos)
admin.site.register(Images)
admin.site.register(ImageDetails)