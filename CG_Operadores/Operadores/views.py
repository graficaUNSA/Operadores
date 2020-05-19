from django.shortcuts import render
from django.http import HttpResponse
from pprint import pprint


# Create your views here.

def index(request):

    # pprint(dir(request))
    return render(request, 'Pages/index.html')
