from django.urls import path

from .views import (
    ClientCsrfView,
    ClientLoginView,
    ClientLogoutView,
    ClientMeView,
    ClientSignupView,
)

urlpatterns = [
    path("csrf", ClientCsrfView.as_view(), name="csrf"),
    path("signup", ClientSignupView.as_view(), name="signup"),
    path("login", ClientLoginView.as_view(), name="login"),
    path("logout", ClientLogoutView.as_view(), name="logout"),
    path("me", ClientMeView.as_view(), name="me"),
]
