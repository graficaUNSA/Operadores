from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from CG_Operadores.settings import MEDIA_ROOT
import os
import shutil

# Create your views here.


def index(request):
    if request.method == 'POST':
        my_file = request.FILES['Original']
        nombre = (os.path.splitext(my_file.name))[0]
        ubication_image = MEDIA_ROOT + "/"+nombre
        if os.path.isdir(ubication_image):
            shutil.rmtree(ubication_image)
        os.mkdir(ubication_image)
        fs = FileSystemStorage(location=ubication_image)
        filename = fs.save(my_file.name, my_file)
        uploaded_file_url = fs.url(filename)
        return redirect(reverse('Ops:operators', kwargs={'name': nombre}))
    return render(request, 'Pages/index.html')


def operators(request, name):

    return render(request, 'Pages/Image_Visualization.html')
