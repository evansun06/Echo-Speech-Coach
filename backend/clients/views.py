from django.contrib.auth import login, logout
from django.middleware.csrf import get_token
from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import LoginSerializer, SignupSerializer, UserSerializer
from .services import authenticate_user, signup_user, to_auth_user_dto


def _auth_user_payload(user) -> dict:
    dto = to_auth_user_dto(user)
    return UserSerializer(dto).data


class ClientSignupView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request: Request) -> Response:
        serializer = SignupSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = signup_user(**serializer.validated_data)
        login(request, user)

        return Response(_auth_user_payload(user), status=status.HTTP_201_CREATED)


class ClientLoginView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request: Request) -> Response:
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = authenticate_user(**serializer.validated_data)
        if user is None:
            return Response(
                {"detail": "Invalid email or password."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        login(request, user)
        return Response(_auth_user_payload(user), status=status.HTTP_200_OK)


class ClientLogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        logout(request)
        return Response(status=status.HTTP_204_NO_CONTENT)


class ClientMeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request: Request) -> Response:
        # Make CSRF token available for SPA POST calls after auth state check.
        get_token(request)
        return Response(_auth_user_payload(request.user), status=status.HTTP_200_OK)


class ClientCsrfView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request: Request) -> Response:
        get_token(request)
        return Response(status=status.HTTP_204_NO_CONTENT)
