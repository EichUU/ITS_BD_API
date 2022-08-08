from django.conf.urls.static import static
from django.urls import path, include
from django.conf import settings

from . import views
from . import datamds_master as dm
from . import modmds_master as modmds
from . import prds

urlpatterns = [
    path('', include('mongoDB.urls')),
    path('', include('hbase.urls')),
    path('', include('analyze.urls')),
    path('getTable', views.get_table, name='get_table'),
    path('hive_query', dm.hive_query, name='hive_query'),
    path('getColumn', views.get_column, name='get_columns'),
    path('insertTable', views.insertTable, name='insertTable'),
    path('getPRDTable', prds.selectPRDTable, name='selectPRDTable'),
    path('insertPRDTable', prds.insertPRDTable, name='insertPRDTable'),
    path('getDataCount', prds.dataCount, name='dataCount'),
    path('query', dm.select_query, name='select_query'),
    path('mongodb_query', dm.mongodb_query, name='mongodb_query'),
    path('usermds', dm.user_mds, name="user_mds"),
    path('file', dm.file_open, name="file_open"),
    path('get', dm.get_data, name="get_data"),
    path('get-csv', dm.get_csv_data, name="get_csv_data"),
    path('keys', dm.get_keys, name="get_keys"),
    path('del', dm.del_one, name="del_one"),
    path('del_many', dm.del_many, name="del_many"),
    path('columns', modmds.get_columns, name="get_columns"),
    path('select_columns', modmds.select_columns, name="select_columns"),
    path('drop', modmds.drop, name="drop"),
    path('change-column', modmds.change_column, name="change_column"),
    path('sort', modmds.sort, name="sort"),
    path('between', modmds.between, name="between"),
    path('abs', modmds.abs_columns, name="abs"),
    path('count', modmds.count_column, name="count"),
    path('describe', modmds.describe_column, name="describe"),
    path('value_count', modmds.value_count, name="value_count"),
    path('or', modmds.get_data_or, name="or"),
    path('strslice', modmds.str_slice, name="strslice"),
    path('decode', modmds.decode, name="decode"),
    path('contain', modmds.contain, name="contain"),
    path('comparison', modmds.comparison, name="comparison"),
    path('isin', modmds.str_in, name="isin"),
    path('rank', modmds.get_data_rank, name="rank"),
    path('group_rank', modmds.group_rank, name="group_rank"),
    path('round', modmds.get_data_round, name="round"),
    path('groupby', modmds.get_data_groupby, name="get_data_groupby"),
    path('groupbyconditiononly', modmds.get_data_groupby_condition, name="groupbyconditiononly"),
    path('groupbyconditionboth', modmds.get_data_groupby_condition_both, name="groupbyconditionboth"),
    path('concat', modmds.concat_mds, name="concat"),
    path('join', modmds.join_mds, name="join"),
    path('merge', modmds.merge_mds, name="merge_mds"),
    path('pivot', modmds.pivot, name="pivot"),
    path('transpose', modmds.transpose, name="transpose"),
    path('to_int', modmds.to_int, name="to_int"),
    path('to_float', modmds.to_float, name="to_float"),
    path('fillna', modmds.fillna, name="fillna"),
    path('calc', modmds.calculator, name="calc"),
    path('wordcloud', modmds.wordcloud, name="wordcloud"),
    path('pie', modmds.get_chart_pie, name="pie"),
    path('csv-download', modmds.download, name="csv-download"),
    path('excel-download', modmds.download_excel, name="excel-download")
]

urlpatterns += static('media', ducument_root=settings.MEDIA_URL)