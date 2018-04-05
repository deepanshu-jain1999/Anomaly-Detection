from rest_framework import serializers
# from .models import File


class FileSerializer(serializers.Serializer):
    csv_file = serializers.FileField()

    class Meta:
        fields = ('csv_file',)



