from django.urls import path
from .views import extract_nic

urlpatterns = [
    path("extract-nic/", extract_nic, name="extract_nic"),
]