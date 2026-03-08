from __future__ import annotations

from dataclasses import dataclass

from django.contrib.auth import authenticate
from django.contrib.auth.models import User


@dataclass(frozen=True)
class AuthUserDTO:
    id: int
    email: str
    name: str


def _display_name(user: User) -> str:
    return (user.first_name or "").strip()


def to_auth_user_dto(user: User) -> AuthUserDTO:
    return AuthUserDTO(
        id=user.id,
        email=user.email,
        name=_display_name(user),
    )


def signup_user(*, email: str, password: str, name: str = "") -> User:
    normalized_email = email.strip().lower()
    return User.objects.create_user(
        username=normalized_email,
        email=normalized_email,
        password=password,
        first_name=name.strip(),
    )


def authenticate_user(*, email: str, password: str) -> User | None:
    normalized_email = email.strip().lower()
    return authenticate(username=normalized_email, password=password)
