from rest_framework import serializers
# from .models import File


class FileSerializer(serializers.Serializer):
    csv_file = serializers.FileField()
    state = serializers.CharField(max_length=100)

    class Meta:
        fields = ('csv_file', 'state',)



