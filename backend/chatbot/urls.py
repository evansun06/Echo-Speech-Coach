from django.urls import path

from .views import create_chat_message, get_chat_history, stream_chat_response

urlpatterns = [
    path(
        "sessions/<str:id>/chat/messages",
        create_chat_message,
        name="chat-messages",
    ),
    path(
        "sessions/<str:id>/chat/streams/<str:response_id>",
        stream_chat_response,
        name="chat-stream",
    ),
    path("sessions/<str:id>/chat/history", get_chat_history, name="chat-history"),
]
