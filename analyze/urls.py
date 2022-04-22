import analyze as analyze
from django.urls import path

from . import analyzeMaster as am

urlpatterns = [
    path('analyze', am.analyze, name='analyze'),
]