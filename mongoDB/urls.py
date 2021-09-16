from django.urls import path

from . import mongo_master as monggo

urlpatterns = [
    path('createCollection', monggo.createCollection, name="createCollection"),
    path('createIRCollection', monggo.createIRCollection, name="createIRCollection"),
    path('insertDocument', monggo.documentInsert, name="insertDocument"),
    path('mongoSelect', monggo.mongoSelect, name="mongoSelect"),
    path('mongoDeletChk', monggo.deleteChk, name="deleteChk")
]