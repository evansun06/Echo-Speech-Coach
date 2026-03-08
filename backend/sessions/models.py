from __future__ import annotations

import uuid

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.db import models
from django.db.models import Q
from django.utils import timezone
from django.utils.deconstruct import deconstructible

MAX_VIDEO_FILE_SIZE_BYTES = 50 * 1024 * 1024
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


class CoachOrchestrationRunStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    PROCESSING = "processing", "Processing"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class CoachAgentExecutionStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    PROCESSING = "processing", "Processing"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class CoachAgentKind(models.TextChoices):
    SUBAGENT = "subagent", "Subagent"
    FLAGSHIP_PERIODIC = "flagship_periodic", "Flagship Periodic"
    FLAGSHIP_FINAL = "flagship_final", "Flagship Final"


class LedgerEntryKind(models.TextChoices):
    SUBAGENT_NOTE = "subagent_note", "Subagent Note"
    FLAGSHIP_IMPRESSION = "flagship_impression", "Flagship Impression"
    FLAGSHIP_FINAL = "flagship_final", "Flagship Final"


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


class CoachOrchestrationRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        CoachingSession,
        on_delete=models.CASCADE,
        related_name="coach_runs",
    )
    run_index = models.PositiveIntegerField()
    status = models.CharField(
        max_length=16,
        choices=CoachOrchestrationRunStatus.choices,
        default=CoachOrchestrationRunStatus.QUEUED,
        db_index=True,
    )
    queued_at = models.DateTimeField(default=timezone.now)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    failed_at = models.DateTimeField(null=True, blank=True)
    latest_ledger_sequence = models.PositiveBigIntegerField(default=0)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [
            models.Index(
                fields=["session", "created_at"],
                name="coach_run_sess_created_idx",
            ),
            models.Index(
                fields=["session", "status"],
                name="coach_run_sess_status_idx",
            ),
            models.Index(
                fields=["session", "run_index"],
                name="coach_run_sess_runidx_idx",
            ),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["session", "run_index"],
                name="coach_run_session_idx_uq",
            ),
            models.UniqueConstraint(
                fields=["session"],
                condition=Q(
                    status__in=[
                        CoachOrchestrationRunStatus.QUEUED,
                        CoachOrchestrationRunStatus.PROCESSING,
                    ]
                ),
                name="coach_run_active_uq",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.session_id} run#{self.run_index} ({self.status})"


class CoachAgentExecution(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        CoachOrchestrationRun,
        on_delete=models.CASCADE,
        related_name="agent_executions",
    )
    execution_index = models.PositiveIntegerField()
    agent_kind = models.CharField(max_length=32, choices=CoachAgentKind.choices)
    agent_name = models.CharField(max_length=128)
    status = models.CharField(
        max_length=16,
        choices=CoachAgentExecutionStatus.choices,
        default=CoachAgentExecutionStatus.QUEUED,
    )
    window_start_ms = models.PositiveIntegerField(null=True, blank=True)
    window_end_ms = models.PositiveIntegerField(null=True, blank=True)
    input_seq_from = models.PositiveBigIntegerField(null=True, blank=True)
    input_seq_to = models.PositiveBigIntegerField(null=True, blank=True)
    output_seq_to = models.PositiveBigIntegerField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    failed_at = models.DateTimeField(null=True, blank=True)
    last_heartbeat_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("execution_index", "created_at")
        indexes = [
            models.Index(
                fields=["run", "execution_index"],
                name="coach_agent_run_exec_idx",
            ),
            models.Index(fields=["run", "status"], name="coach_agent_run_status_idx"),
            models.Index(fields=["run", "created_at"], name="coach_agent_run_created_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["run", "execution_index"],
                name="coach_agent_run_exec_uq",
            ),
            models.CheckConstraint(
                name="coach_agent_window_bounds_ck",
                condition=Q(window_start_ms__isnull=True, window_end_ms__isnull=True)
                | Q(
                    window_start_ms__isnull=False,
                    window_end_ms__isnull=False,
                    window_start_ms__lte=models.F("window_end_ms"),
                ),
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id} agent#{self.execution_index} ({self.agent_name})"


class CoachLedgerEntry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        CoachOrchestrationRun,
        on_delete=models.CASCADE,
        related_name="ledger_entries",
    )
    agent_execution = models.ForeignKey(
        CoachAgentExecution,
        on_delete=models.SET_NULL,
        related_name="ledger_entries",
        null=True,
        blank=True,
    )
    sequence = models.PositiveBigIntegerField()
    entry_kind = models.CharField(max_length=32, choices=LedgerEntryKind.choices)
    agent_kind = models.CharField(
        max_length=32,
        choices=CoachAgentKind.choices,
        null=True,
        blank=True,
    )
    agent_name = models.CharField(max_length=128, blank=True, default="")
    window_start_ms = models.PositiveIntegerField(null=True, blank=True)
    window_end_ms = models.PositiveIntegerField(null=True, blank=True)
    content = models.TextField()
    payload = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("sequence", "created_at")
        indexes = [
            models.Index(fields=["run", "sequence"], name="coach_ledger_run_seq_idx"),
            models.Index(fields=["run", "created_at"], name="coach_ledger_run_created_idx"),
            models.Index(fields=["run", "entry_kind"], name="coach_ledger_run_kind_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["run", "sequence"],
                name="coach_ledger_run_seq_uq",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id} seq={self.sequence} ({self.entry_kind})"


class SessionOverallMetrics(models.Model):
    session = models.OneToOneField(
        CoachingSession,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="overall_metrics_row",
        db_column="session_uuid",
    )
    metrics = models.JSONField()

    class Meta:
        db_table = "session_overall_metrics"

    def __str__(self) -> str:
        return f"{self.session_id} overall_metrics"


class SessionAlignedWord(models.Model):
    session = models.ForeignKey(
        CoachingSession,
        on_delete=models.CASCADE,
        related_name="aligned_words",
        db_column="session_uuid",
    )
    word_id = models.IntegerField()
    pk = models.CompositePrimaryKey("session", "word_id")
    word_json = models.JSONField()
    start_time = models.FloatField()
    end_time = models.FloatField()

    class Meta:
        db_table = "session_aligned_words"
        indexes = [
            models.Index(
                fields=["session", "start_time"],
                name="sess_align_s_start_idx",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.session_id} word#{self.word_id}"


class SessionEvent(models.Model):
    session = models.ForeignKey(
        CoachingSession,
        on_delete=models.CASCADE,
        related_name="events",
        db_column="session_uuid",
    )
    event_id = models.IntegerField()
    pk = models.CompositePrimaryKey("session", "event_id")
    event_json = models.JSONField()
    start_time = models.FloatField()
    end_time = models.FloatField()

    class Meta:
        db_table = "session_events"
        indexes = [
            models.Index(
                fields=["session", "start_time"],
                name="sess_event_s_start_idx",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.session_id} event#{self.event_id}"
