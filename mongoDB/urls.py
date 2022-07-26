from django.urls import path

from . import mongo_master as monggo
from . import BDPA020

urlpatterns = [
    path('createCollection', monggo.createCollection, name="createCollection"),
    path('createIRCollection', monggo.createIRCollection, name="createIRCollection"),
    path('insertDocument', monggo.documentInsert, name="insertDocument"),
    path('mongoSelect', monggo.mongoSelect, name="mongoSelect"),
    path('mongoDeletChk', monggo.deleteChk, name="deleteChk"),
    path('getStdInform', BDPA020.getStdInform, name="getStdInform"),
    path('getStdGraph', BDPA020.getImage, name="getStdInform"),
    path('getColumnComment', BDPA020.getColumnComment, name="getColumnComment"),
    path('getMongoQuery', BDPA020.getMongoQuery, name="getColumnComment"),
]