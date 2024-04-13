from django.shortcuts import render

def home(request):
    return render(request,'Home.html')


def diabetes(request):
    return