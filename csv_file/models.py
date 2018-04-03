from django.db import models


class File(models.Model):
    csv_file = models.FileField(upload_to='file/', blank=False, null=False)
    title = models.CharField(max_length=20)