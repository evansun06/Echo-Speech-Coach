from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response


@extend_schema(tags=["chat"])
@api_view(["POST"])
def create_chat_message(request: Request, id: str) -> Response:
    """Create a new chat message for the specified session."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["chat"])
@api_view(["GET"])
def stream_chat_response(request: Request, id: str, response_id: str) -> Response:
    """Stream chat response events for a previously created response."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["chat"])
@api_view(["GET"])
def get_chat_history(request: Request, id: str) -> Response:
    """Return prior chat messages for the specified session."""
    return Response({}, status=status.HTTP_200_OK)
