from rest_framework import serializers
from .models import File


class FileSerializer(serializers.ModelSerializer):
    csv_file = serializers.FileField()
    class Meta:
        model = File
        fields = ('csv_file', 'title')



