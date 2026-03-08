from __future__ import annotations

from rest_framework import serializers

from .models import CoachingSession


def _absolute_file_url(request, file_field) -> str | None:
    if not file_field:
        return None

    try:
        url = file_field.url
    except ValueError:
        return None

    if request is None:
        return url
    return request.build_absolute_uri(url)


class CreateSessionSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, max_length=255)

    def validate_title(self, value: str) -> str:
        title = value.strip()
        if not title:
            raise serializers.ValidationError("Title cannot be blank.")
        return title


class UploadSessionVideoSerializer(serializers.Serializer):
    video_file = serializers.FileField(required=True)


class UploadSessionAssetsSerializer(serializers.Serializer):
    supplementary_pdf_1 = serializers.FileField(required=False)
    supplementary_pdf_2 = serializers.FileField(required=False)
    supplementary_pdf_3 = serializers.FileField(required=False)
    speaker_context = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        update_fields = {
            "supplementary_pdf_1",
            "supplementary_pdf_2",
            "supplementary_pdf_3",
            "speaker_context",
        }
        if not any(field in self.initial_data for field in update_fields):
            raise serializers.ValidationError(
                "At least one of supplementary_pdf_1, supplementary_pdf_2, "
                "supplementary_pdf_3, or speaker_context is required."
            )
        return attrs


class SessionListItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = CoachingSession
        fields = (
            "id",
            "title",
            "status",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class SessionDetailSerializer(serializers.ModelSerializer):
    video_file_url = serializers.SerializerMethodField()
    supplementary_pdf_1_url = serializers.SerializerMethodField()
    supplementary_pdf_2_url = serializers.SerializerMethodField()
    supplementary_pdf_3_url = serializers.SerializerMethodField()

    class Meta:
        model = CoachingSession
        fields = (
            "id",
            "title",
            "status",
            "created_at",
            "updated_at",
            "video_file_url",
            "supplementary_pdf_1_url",
            "supplementary_pdf_2_url",
            "supplementary_pdf_3_url",
            "speaker_context",
        )
        read_only_fields = fields

    def get_video_file_url(self, obj: CoachingSession) -> str | None:
        request = self.context.get("request")
        return _absolute_file_url(request, obj.video_file)

    def get_supplementary_pdf_1_url(self, obj: CoachingSession) -> str | None:
        request = self.context.get("request")
        return _absolute_file_url(request, obj.supplementary_pdf_1)

    def get_supplementary_pdf_2_url(self, obj: CoachingSession) -> str | None:
        request = self.context.get("request")
        return _absolute_file_url(request, obj.supplementary_pdf_2)

    def get_supplementary_pdf_3_url(self, obj: CoachingSession) -> str | None:
        request = self.context.get("request")
        return _absolute_file_url(request, obj.supplementary_pdf_3)
