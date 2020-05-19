from django.urls import path
from . import views

app_name = 'Operations'

urlpatterns = [
    path('', views.index, name='index'),
]
