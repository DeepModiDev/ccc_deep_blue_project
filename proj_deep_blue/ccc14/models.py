from django.db import models
from django.contrib.auth.models import User

class Images(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.FileField(upload_to='images/')
    date = models.DateTimeField(auto_now_add=False,blank=True,auto_now=False)
    imageTitle = models.CharField(max_length=200)
    
    class Meta:
        verbose_name = 'Image'
        verbose_name_plural = 'Images'

    def __str__(self):
        return self.imageTitle

    def delete(self,*args,**kwargs):
        self.image.delete()
        super().delete(*args,**kwargs)

class Videos(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.FileField(upload_to='videos/')

    class Meta:
        verbose_name = 'video'
        verbose_name_plural = 'videos'

    def __str__(self):
        return self.video.name

    def delete(self,*args,**kwargs):
        self.video.delete()
        super().delete(*args,**kwargs)

class DetectionVideos(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.FileField(upload_to='videos/detections/')
    thumbnail = models.FileField(upload_to='videos/detections/thumbnails/')
    date = models.DateTimeField(auto_now_add=False,blank=True,auto_now=False)

    class Meta:
        verbose_name = 'Detection Video'
        verbose_name_plural = 'Detection Videos'

    def __str__(self):
        return self.video.name

    def delete(self,*args,**kwargs):
        self.video.delete()
        super().delete(*args,**kwargs)

class ImageDetails(models.Model):
    image = models.ForeignKey(Images,on_delete=models.CASCADE)
    imageDetails = models.CharField(max_length=200)

    class Meta:
        verbose_name = 'Image Detail'
        verbose_name_plural = 'Images Details'

    def __str__(self):
        return self.image.image.name.split('/')[1]