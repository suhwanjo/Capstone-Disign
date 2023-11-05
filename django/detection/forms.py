from django import forms
from .models import UpImage, UpVideo

class UpImageForm(forms.ModelForm):
    class Meta:
        model = UpImage
        fields = ['image']


class UpVideoForm(forms.ModelForm):
    class Meta:
        model = UpVideo
        fields = ['video']


