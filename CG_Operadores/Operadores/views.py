from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from CG_Operadores.settings import MEDIA_ROOT
from django.views.decorators.csrf import csrf_protect
import os
import shutil
from .utils import *
# Create your views here.

def get_original_file_extra(path):
    return [obj for obj in os.listdir(path) if os.path.isfile(path + "/" + obj)][0]


def index(request):
    if request.method == 'POST':
        my_file = request.FILES['Original']
        archivo = (os.path.splitext(my_file.name))
        nombre = archivo[0]
        ubication_image = MEDIA_ROOT + "/"+nombre
        if os.path.isdir(ubication_image):
            shutil.rmtree(ubication_image)
        os.mkdir(ubication_image)
        fs = FileSystemStorage(location=ubication_image)
        filename = fs.save(my_file.name, my_file)
        return redirect(reverse('Ops:operators', kwargs={'name': nombre}))
    return render(request, 'Pages/index.html')


def image_exponential(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        constante1 = float(request.POST['constante1'])
        solve(path, values, 1, constante, constante1)
        return JsonResponse({'Estado': 'Works'})
    else:
        return JsonResponse({'Estado': 'fallo'})


def operators(request, name):
    path = MEDIA_ROOT + "/" + name
    values = get_original_file_extra(path)
    send_page = {}
    send_page['imagen'] = "/media/" + name + "/" + values
    send_page['camino'] = path
    send_page['nombre'] = values
    # send_page['imagen_exponential'] = "/media/" + name + "/" + "exponential.png"
    return render(request, 'Pages/Image_Visualization.html', send_page)


