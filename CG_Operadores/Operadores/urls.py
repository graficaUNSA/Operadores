from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


app_name = 'Ops'

urlpatterns = [
    path('', views.index, name='index'),
    path('img/<str:name>', views.operators, name='operators'),
    path('img/<str:name>/data_thresholding', views.image_thresholding),
    path('img/<str:name>/data_contrast', views.image_contrast_streching),
    path('img/<str:name>/data_equalization', views.image_exponential),
    path('img/<str:name>/data_exponential', views.image_exponential),
    path('img/<str:name>/data_logarithm', views.image_logarithm),
    path('img/<str:name>/data_square', views.image_square_root),
    path('img/<str:name>/data_pow', views.image_raise_power),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)