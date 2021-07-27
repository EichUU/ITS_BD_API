from django.urls import path

from . import views
from . import datamds_master as dm

urlpatterns = [
    path('getTable', views.get_table, name='get_table'),
    path('getColumn', views.get_column, name='get_columns'),
    path('insertTable', views.insertTable, name='insertTable'),
    path('query', dm.select_query, name='select_query'),
    path('usermds', dm.user_mds, name="user_mds"),
    path('file', dm.file_open, name="file_open"),
    path('get', dm.get_data, name="get_data"),
    path('get-csv', dm.get_csv_data, name="get_csv_data"),
    path('keys', dm.get_keys, name="get_keys"),
    path('del', dm.del_one, name="del_one"),
    path('del_many', dm.del_many, name="del_many")
]