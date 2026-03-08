from django.urls import path

from .views import (
    get_session,
    get_session_chat_context,
    get_session_timeline,
    get_session_video_stream,
    sessions_collection,
    start_session_analysis,
    upload_session_assets,
    upload_session_video,
)

urlpatterns = [
    path("sessions", sessions_collection, name="sessions-collection"),
    path("sessions/<str:id>/video", upload_session_video, name="session-video"),
    path("sessions/<str:id>/assets", upload_session_assets, name="session-assets"),
    path(
        "sessions/<str:id>/start-analysis",
        start_session_analysis,
        name="session-start-analysis",
    ),
    path("sessions/<str:id>", get_session, name="session-detail"),
    path("sessions/<str:id>/timeline", get_session_timeline, name="session-timeline"),
    path(
        "sessions/<str:id>/chat-context",
        get_session_chat_context,
        name="session-chat-context",
    ),
    path(
        "sessions/<str:id>/video-stream",
        get_session_video_stream,
        name="session-video-stream",
    ),
]
