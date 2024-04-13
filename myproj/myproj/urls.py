from django.contrib import admin
from django.urls import path
from model import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home),
    path('diabetes',views.diabetes)
]
