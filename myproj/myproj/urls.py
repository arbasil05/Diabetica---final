from django.contrib import admin
from django.conf.urls.static import static
from django.urls import path
from model import views
from django.conf import settings
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home),
    path('home',views.home),
    path('diabetes',views.diabetes),
    path('choice',views.choice),
    path('forms',views.forms),
    path('demo',views.demo),
    path('about',views.about),
    path('contact',views.contact),
    #path('send_email',views.send_email)
]

if settings.DEBUG:
    urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
