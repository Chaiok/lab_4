from django.db import models
import os
from django.utils.translation import gettext_lazy as _
from .validators import validate_file_extension

class ImageModel(models.Model):
    image = models.FileField(_("Файл"), upload_to='images', validators=[validate_file_extension])

    class Meta:
        verbose_name = "Image"
        verbose_name_plural = "Images"

    def __str__(self):
        return str(os.path.split(self.image.path)[-1])
    
    def filename(self):
        return os.path.basename(self.image.name)