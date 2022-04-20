from django.urls import path

from . import hbase_master as hbase

urlpatterns = [
    path('hbaseInsert', hbase.hbaseInsert, name='hbaseInsert'),
]