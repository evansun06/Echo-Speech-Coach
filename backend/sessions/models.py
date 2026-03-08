from __future__ import annotations

import uuid

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.db import models
from django.db.models import Q
from django.utils.deconstruct import deconstructible

MAX_VIDEO_FILE_SIZE_BYTES = 500 * 1024 * 1024
MAX_SUPPLEMENTARY_PDF_FILE_SIZE_BYTES = 25 * 1024 * 1024


@deconstructible
class MaxFileSizeValidator:
    def __init__(self, *, max_bytes: int, label: str) -> None:
        self.max_bytes = max_bytes
        self.label = label

    def __call__(self, value) -> None:
        if not value:
            return

        size = getattr(value, "size", None)
        if size is None:
            return

        if size > self.max_bytes:
            raise ValidationError(
                f"{self.label} exceeds {self.max_bytes} bytes."
            )


class SessionStatus(models.TextChoices):
    DRAFT = "draft", "Draft"
    MEDIA_ATTACHED = "media_attached", "Media Attached"
    QUEUED_ML = "queued_ml", "Queued ML"
    PROCESSING_ML = "processing_ml", "Processing ML"
    ML_READY = "ml_ready", "ML Ready"
    PROCESSING_COACH = "processing_coach", "Processing Coach"
    READY = "ready", "Ready"
    COACH_FAILED = "coach_failed", "Coach Failed"
    FAILED = "failed", "Failed"


class CoachingSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="coaching_sessions",
    )
    title = models.CharField(max_length=255, default="Untitled Session")
    status = models.CharField(
        max_length=32,
        choices=SessionStatus.choices,
        default=SessionStatus.DRAFT,
        db_index=True,
    )

    video_file = models.FileField(
        upload_to="sessions/videos/%Y/%m/%d/",
        max_length=512,
        null=True,
        blank=True,
        validators=[
            FileExtensionValidator(allowed_extensions=["mp4", "webm"]),
            MaxFileSizeValidator(
                max_bytes=MAX_VIDEO_FILE_SIZE_BYTES,
                label="Video file",
            ),
        ],
    )
    supplementary_pdf_1 = models.FileField(
        upload_to="sessions/assets/%Y/%m/%d/",
        max_length=512,
        null=True,
        blank=True,
        validators=[
            FileExtensionValidator(allowed_extensions=["pdf"]),
            MaxFileSizeValidator(
                max_bytes=MAX_SUPPLEMENTARY_PDF_FILE_SIZE_BYTES,
                label="Supplementary PDF",
            ),
        ],
    )
    supplementary_pdf_2 = models.FileField(
        upload_to="sessions/assets/%Y/%m/%d/",
        max_length=512,
        null=True,
        blank=True,
        validators=[
            FileExtensionValidator(allowed_extensions=["pdf"]),
            MaxFileSizeValidator(
                max_bytes=MAX_SUPPLEMENTARY_PDF_FILE_SIZE_BYTES,
                label="Supplementary PDF",
            ),
        ],
    )
    supplementary_pdf_3 = models.FileField(
        upload_to="sessions/assets/%Y/%m/%d/",
        max_length=512,
        null=True,
        blank=True,
        validators=[
            FileExtensionValidator(allowed_extensions=["pdf"]),
            MaxFileSizeValidator(
                max_bytes=MAX_SUPPLEMENTARY_PDF_FILE_SIZE_BYTES,
                label="Supplementary PDF",
            ),
        ],
    )
    speaker_context = models.TextField(blank=True, default="")
    ml_task_id = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    coach_task_id = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [
            models.Index(fields=["user", "created_at"], name="coach_sess_user_created_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                name="coach_sess_non_draft_has_video",
                condition=Q(status=SessionStatus.DRAFT)
                | (Q(video_file__isnull=False) & ~Q(video_file="")),
            ),
        ]

    def __str__(self) -> str:
        return f"{self.title} ({self.id}, {self.status})"
