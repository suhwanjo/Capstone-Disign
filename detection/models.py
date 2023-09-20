from django.db import models

# Create your models here.

class UpImage(models.Model):
    image = models.ImageField(upload_to='upimg/', blank=False, null=False)
    email = models.EmailField(max_length=128, null=True)

class UpVideo(models.Model):
    video = models.FileField(upload_to='upvideo/', blank=False, null=False)
    email = models.EmailField(max_length=128, null=True)