from django.urls import path

from . import mongo_master as monggo

urlpatterns = [
    path('createIRCollection', monggo.createIRCollection, name="createIRCollection")
]