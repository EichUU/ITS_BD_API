from django.urls import path

from . import mongo_master as monggo

urlpatterns = [
    path('createCollection', monggo.createCollection, name="createCollection"),
    path('createIRCollection', monggo.createIRCollection, name="createIRCollection")
]