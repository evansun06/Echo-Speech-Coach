from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response


@extend_schema(tags=["sessions"])
def create_session(request: Request) -> Response:
    """Create a new coaching session."""
    return Response({}, status=status.HTTP_201_CREATED)


@extend_schema(tags=["sessions"])
def list_sessions(request: Request) -> Response:
    """List coaching sessions with status metadata."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema_view(
    get=extend_schema(tags=["sessions"]),
    post=extend_schema(tags=["sessions"]),
)
@api_view(["GET", "POST"])
def sessions_collection(request: Request) -> Response:
    """Dispatch GET/POST requests for the sessions collection route."""
    if request.method == "POST":
        return create_session(request)
    return list_sessions(request)


@extend_schema(tags=["sessions"])
@api_view(["POST"])
def upload_session_video(request: Request, id: str) -> Response:
    """Attach or upload a video file for the given session."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["POST"])
def upload_session_assets(request: Request, id: str) -> Response:
    """Attach optional assets such as script or slides to a session."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["POST"])
def start_session_analysis(request: Request, id: str) -> Response:
    """Start asynchronous analysis for the specified session."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session(request: Request, id: str) -> Response:
    """Fetch a single session, including coach progress details."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session_timeline(request: Request, id: str) -> Response:
    """Return timeline events generated for a session."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session_chat_context(request: Request, id: str) -> Response:
    """Return prepared chat context for a session."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session_video_stream(request: Request, id: str) -> Response:
    """Stream the session video resource for playback."""
    return Response({}, status=status.HTTP_200_OK)
