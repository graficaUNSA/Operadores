from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse
from pprint import pprint
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Create your views here.


def index(request):
    if request.method == 'POST':
        myfile = request.FILES['Original']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return redirect(reverse('Ops:operators', kwargs={'name': filename}))
    return render(request, 'Pages/index.html')


def operators(request, name):
    print(name)
    return render(request, 'Pages/Image_Visualization.html')
