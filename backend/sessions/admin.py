from django.contrib import admin

from .models import (
    CoachAgentExecution,
    CoachLedgerEntry,
    CoachOrchestrationRun,
    CoachingSession,
    SessionOverallMetrics,
)


@admin.register(CoachingSession)
class CoachingSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "user", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("id", "title", "user__email")


@admin.register(CoachOrchestrationRun)
class CoachOrchestrationRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "session",
        "run_index",
        "status",
        "latest_ledger_sequence",
        "created_at",
    )
    list_filter = ("status", "created_at")
    search_fields = ("id", "session__id")


@admin.register(CoachAgentExecution)
class CoachAgentExecutionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "run",
        "execution_index",
        "agent_kind",
        "agent_name",
        "status",
        "started_at",
    )
    list_filter = ("status", "agent_kind", "created_at")
    search_fields = ("id", "agent_name", "run__id")


@admin.register(CoachLedgerEntry)
class CoachLedgerEntryAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "run",
        "sequence",
        "entry_kind",
        "agent_kind",
        "agent_name",
        "created_at",
    )
    list_filter = ("entry_kind", "agent_kind", "created_at")
    search_fields = ("id", "run__id", "agent_name", "content")


@admin.register(SessionOverallMetrics)
class SessionOverallMetricsAdmin(admin.ModelAdmin):
    list_display = ("session",)
    search_fields = ("session__id",)
