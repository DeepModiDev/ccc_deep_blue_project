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
    videoTitle = models.CharField(max_length=200)
    video = models.FileField(upload_to='videos/detections/')
    thumbnail = models.FileField(upload_to='videos/detections/thumbnails/')
    date = models.DateTimeField(auto_now_add=False,blank=True,auto_now=False)
    min_count = models.IntegerField(blank=True,null=True)
    max_count = models.IntegerField(blank=True,null=True)
    average_count = models.IntegerField(blank=True,null=True)
    median_count = models.IntegerField(blank=True,null=True)

    class Meta:
        verbose_name = 'Detection Video'
        verbose_name_plural = 'Detection Videos'

    def __str__(self):
        return self.videoTitle

    def delete(self,*args,**kwargs):
        self.video.delete()
        self.thumbnail.delete()
        super().delete(*args,**kwargs)

class ImageDetails(models.Model):
    image = models.ForeignKey(Images,on_delete=models.CASCADE)
    imageDetails = models.CharField(max_length=200)

    class Meta:
        verbose_name = 'Image Detail'
        verbose_name_plural = 'Images Details'

    def __str__(self):
        return self.image.image.name.split('/')[1]

class MallOwnerShopOwner(models.Model):
    mallOwner = models.ForeignKey(User,on_delete=models.CASCADE,related_name='mall_owner_foreign_key')
    shopOwner = models.ForeignKey(User,on_delete=models.CASCADE,related_name='shop_owner_foreign_key')

    class Meta:
        verbose_name = "Mall Owner and Shop Owner's relation key"
        verbose_name_plural = "Mall Owner and Shop Owner's relation keys"

    def __str__(self):
        return "MallOwner: "+str(self.mallOwner.pk)+" ShopOwner: "+str(self.shopOwner.pk)
