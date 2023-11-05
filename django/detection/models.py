from django.db import models
# from django.utils import timezone

# Create your models here.
class UpImage(models.Model):
    #img_file_name = models.CharField(default="default", max_length=50, primary_key=True)
    #upload_date = models.DateTimeField(default=default_value, auto_created=True)
    image = models.ImageField(upload_to='upimg/', blank=False, null=False)

"""
class Image_result(models.Model):
    is_fake = models.SmallIntegerField(primary_key=True)
    img_file_name = models.ForeignKey("UpImage", related_name="image", on_delete=models.CASCADE, db_column="img_file_name")
"""

class UpVideo(models.Model):
    #vid_file_name = models.CharField(default="default", max_length=50, primary_key=True)
    #upload_date = models.DateTimeField(default=default_value, auto_created=True)
    video = models.FileField(upload_to='upvid/', blank=False, null=False)
