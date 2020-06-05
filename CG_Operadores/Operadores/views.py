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
        estado, name_image, npath = solve_exponential(path, values, constante, constante1)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_logarithm(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        estado, name_image, npath = solve_logarithm(path, values, constante)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_raise_power(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        constante1 = float(request.POST['constante1'])
        estado, name_image, npath = solve_raise_power(path, values, constante, constante1)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_addition_const(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        estado, name_image, npath = solve_addition_constant(path, values, constante)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_difference_const(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        estado, name_image, npath = solve_difference_constant(path, values, constante)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_dot_const(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        estado, name_image, npath = solve_dot_constant(path, values, constante)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_division_const(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        estado, name_image, npath = solve_division_constant(path, values, constante)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_contrast_streching(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        constante1 = float(request.POST['constante1'])
        estado, name_image, npath = solve_contrast_streching(path, values, constante, constante1)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_square_root(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        estado, name_image, npath = solve_square_root(path, values, constante)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_thresholding(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constante = float(request.POST['constante'])
        constante1 = float(request.POST['constante1'])
        estado, name_image, npath = solve_thresholding(path, values, constante, constante1)
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def image_histogram_equalization(request, name):
    if request.method == 'POST':
        path = request.POST['camino']
        values = request.POST['nombre']
        constanteX_1 = int(request.POST['constanteX_1'])
        constanteY_1 = int(request.POST['constanteY_1'])
        constanteX_2 = int(request.POST['constanteX_2'])
        constanteY_2 = int(request.POST['constanteY_2'])
        estado, name_image, npath = solve_histogram_equalization(path, values, (constanteX_1, constanteY_1), (constanteX_2, constanteY_2))
        if estado:
            return JsonResponse({
                'Estado': "OK",
                'nombre': name_image,
                'camino': npath,
                'imagen': "/media/" + name_image + "/" + name_image
            })
        else:
            return JsonResponse({'State': 'fail'})
    else:
        return JsonResponse({'State': 'fail'})


def operators(request, name):
    path = MEDIA_ROOT + "/" + name
    values = get_original_file_extra(path, name)
    send_page = {}
    send_page['imagen'] = "/media/" + name + "/" + values
    send_page['camino'] = path
    send_page['nombre'] = values
    # send_page['imagen_exponential'] = "/media/" + name + "/" + "exponential.png"
    return render(request, 'Pages/Image_Visualization.html', send_page)

