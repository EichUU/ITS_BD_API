from django.urls import path

from . import views

urlpatterns = [
    path('getTable', views.get_table, name='get_table'),
    path('getColumn', views.get_column, name='get_columns'),
    path('insertTable', views.insertTable, name='insertTable'),
]