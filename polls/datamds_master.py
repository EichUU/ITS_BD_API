# This is a sample Python script.
from io import StringIO
import cx_Oracle
# from gevent.pywsgi import WSGIServer
from pyhive import hive
import pandas as pd
import pyarrow as pa
import redis
import uuid
import json
import logging
import time
import bson
import pql

from pandas.io.sql import DatabaseError
from django.http import HttpResponse

import datetime as dt
from urllib.parse import unquote
from sqlalchemy import create_engine

########################################################################################################################
# docker tag : seokjoo/datamds
# version 1.6.9
# msg : 1.5.5 COCD change from gongt to thaksa
# msg : 1.5.6 usermds change to receive data.
# msg : 1.5.7 Change exception message to code.
# msg : 1.5.8 add jsontoredis function
# msg : 1.5.9 bug fix query -> _NM duplicate issue change _KNM
# msg : 1.6.0 query - 원래 컬럼 리스트 -> 들어온 순서대로 reindex 하는 부분 추가
# msg : 1.6.1 (1.6.0)의 컬럼 순서 바꾸는 부분 받은 인덱스에 원래 인덱스 붙여서 duplicate drop 하는 방식으로 변경
# msg : 1.6.2 get 의 데이터 가져오는 부분에서 float 을 to_json 하는 과정에서 부동소수점 오류가 나옴 -> str로 고쳤다가 다시 float으로
# msg : 1.6.3 get bug fix
# msg : 1.6.4 bug fix
# msg : 1.6.5 query get datetime data to string type
# msg : 1.6.6 hive query added
# msg : 1.6.7 bug fix requirements.txt
# msg : 1.6.8 bug fix
# msg : 1.6.9 bug fix
########################################################################################################################
# pool = cx_Oracle.SessionPool("HAKSA", "!#HAKSA*",
#                             "192.168.102.70:1521/SMISBK", min=2, max=10, increment=1, threaded=True, encoding="UTF-8")

# create logger
from werkzeug.sansio.response import Response

logger = logging.getLogger('mod')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

INT_TYPE = {"int_", "int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
FLOAT_TYPE = {"float_", "float", "float16", "float32", "float64"}

# test => host='203.247.194.215', port=61379, db=0
# deploy => host='10.111.82.182', port=6379, db=0
r_pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
# r_pool = redis.ConnectionPool(host='203.247.194.215', port=61379, db=0)
from dbcon.connDB import get_conn
from pymongo import MongoClient
from bson.json_util import dumps, loads
from bson.objectid import ObjectId

# HIVE CONNECTION INFO
# host = "203.247.194.212"
host = "192.168.0.193"
port = 10000

# MongDB CONNECTION INFO
mongo_host = "192.168.0.193"
mongo_port = 20000

# expire_time = 2 : 2분 주기로 connection check(ping)

connection_tibero = get_conn("tibero", "TIBERO", "UPMS", "UPMS12#$", "", "")
#connection_hive = get_conn("hive", "", "", "", "192.168.0.100", 10000)
# connection = cx_Oracle.connect("HAKSA", "!#HAKSA*", "192.168.102.70:1521/SMISBK?expire_time=2")
# connection.callTimeout = 0
# gongt_connection = cx_Oracle.connect("GONGT", "!#GONGT*", "192.168.102.70:1521/SMISBK?expire_time=2")
# gongt_connection.callTimeout = 0
# ghaksa_connection = cx_Oracle.connect("GHAKSA", "!#GHAKSA*", "192.168.102.70:1521/SMISBK?expire_time=2")
# ghaksa_connection.callTimeout = 0
# hangj_connection = cx_Oracle.connect("HANGJ", "!#HANGJ*", "192.168.102.70:1521/SMISBK?expire_time=2")
# hangj_connection.callTimeout = 0
# resch_connection = cx_Oracle.connect("RESCH", "!#RESCH*", "192.168.102.70:1521/SMISBK?expire_time=2")
# resch_connection.callTimeout = 0
# buseol_connection = cx_Oracle.connect("BUSEOL", "!#BUSEOL*", "192.168.102.70:1521/SMISBK?expire_time=2")
# buseol_connection.callTimeout = 0
# abeek_connection = cx_Oracle.connect("ABEEK", "!#ABEEK*", "192.168.102.70:1521/SMISBK?expire_time=2")
# abeek_connection.callTimeout = 0
# lifedu_connection = cx_Oracle.connect("LIFEDU", "!#LIFEDU*", "192.168.102.70:1521/SMISBK?expire_time=2")
# lifedu_connection.callTimeout = 0
# cary_connection = cx_Oracle.connect("CARY", "!#CARY*", "192.168.102.70:1521/SMISBK?expire_time=2")
# cary_connection.callTimeout = 0
# stats_connection = cx_Oracle.connect("STATS", "!#STATS*", "192.168.102.70:1521/SMISBK?expire_time=2")
# stats_connection.callTimeout = 0
# thaksa_connection = cx_Oracle.connect("thaksa_new", "THAKSA", "203.247.194.209:1521/XE?expire_time=2")
# thaksa_connection.callTimeout = 0

# connection with sqlalchemy : better than cx_oracle ? (test need)
# connection = create_engine('oracle://HAKSA:!#HAKSA*@192.168.102.70:1521/SMISBK',
#                            pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# gongt_connection = create_engine('oracle://GONGT:!#GONGT*@192.168.102.70:1521/SMISBK',
#                                  pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# ghaksa_connection = create_engine('oracle://GHAKSA:!#GHAKSA*@192.168.102.70:1521/SMISBK',
#                                   pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# hangj_connection = create_engine('oracle://HANGJ:!#HANGJ*@192.168.102.70:1521/SMISBK',
#                                  pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# resch_connection = create_engine('oracle://RESCH:!#RESCH*@192.168.102.70:1521/SMISBK',
#                                  pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# buseol_connection = create_engine('oracle://BUSEOL:!#BUSEOL*@192.168.102.70:1521/SMISBK',
#                                   pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# abeek_connection = create_engine('oracle://ABEEK:!#ABEEK*@192.168.102.70:1521/SMISBK',
#                                  pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# lifedu_connection = create_engine('oracle://LIFEDU:!#LIFEDU*@192.168.102.70:1521/SMISBK',
#                                   pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# cary_connection = create_engine('oracle://CARY:!#CARY*@192.168.102.70:1521/SMISBK',
#                                 pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# stats_connection = create_engine('oracle://STATS:!#STATS*@192.168.102.70:1521/SMISBK',
#                                  pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)
# thaksa_connection = create_engine('oracle://thaksa_new:THAKSA@203.247.194.209:1521/XE',
#                                   pool_recycle=500, pool_size=10, max_overflow=20, echo_pool=True)

# web DB connection
# web_connection = cx_Oracle.connect("tcloud", "tcloud", "203.247.194.209:1521/XE?expire_time=2")

# 업로드 허용되는 파일 형식
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}


def get_uuid():
    uid = uuid.uuid1()
    uid = str(uid).replace("-", "_")
    return uid



# def free_df(df: pd.DataFrame):
#     del [[df]]
#     gc.collect()
#     df = pd.DataFrame()


def get_data_from_redis(key):
    print("get_data_from_redis====" + key)
    r = redis.Redis(connection_pool=r_pool)
    try:
        # context = pa.default_serialization_context()
        print("get_data_from_redis====11111")
        data = pa.deserialize(r.get(key))
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print("get_data_from_redis====2222222")
        logger.debug(e)
        logger.debug("redis key has no data or not exist.")
        return None


def set_data_to_redis(df):
    print(r_pool)
    r = redis.Redis(connection_pool=r_pool)
    print(1)
    # context = pa.default_serialization_context()
    uuid1 = get_uuid()
    print(2)
    r.set(uuid1, pa.serialize(df).to_buffer().to_pybytes())
    print(3)
    return uuid1



# def check_db():
#     ysql = 'select 0 from dual'
#     try:



def json_to_redis(request):
    try:
        key = request.args.get("key", default=None, type=str)
        value = request.args.get("value", default=None, type=str)
        df = pd.DataFrame(json.loads(value))
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        r = redis.Redis(connection_pool=r_pool)
        r.set(key, pa.serialize(df).to_buffer().to_pybytes())
        return str(key), 200
    except Exception as e:
        print(e)
        resp = HttpResponse("file not correct")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp



def hive_query(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        start = time.time()
        query = body['query']
        name = body['name']
        columns = body['columns']
        cocd_columns = body['cocdcol']
        cocd_code = body['codes']
        exclusion = body['exclusion']
        query_split = query.split("ROWNUM")
    else:
        start = time.time()
        query = request.GET.get("query")
        name = request.GET.get("name")
        columns = request.GET.get("columns")
        cocd_columns = request.GET.get("cocdcol")
        cocd_code = request.GET.get("codes")
        exclusion = request.GET.get("exclusion")
        query_split = query.split("ROWNUM")
    if len(query_split) > 1:
        l_query = "".join(query_split[0].rsplit("AND", 1))
        query = l_query + "LIMIT" + query_split[-1].replace("<", " ").replace("=", " ")
    print(query)
    # if str(query_split[-1]).contains("ROWNUM"):
    conn = hive.Connection(host=host, port=port, auth="NOSASL", database="test")
    try:
        df = pd.read_sql(query, conn)
    except DatabaseError as e:
        logger.debug(e)
        resp = HttpResponse("exception occurs")
        resp.headers['exception'] = "999051"
        resp.status_code = 400
        return resp
    if df is None or len(df) == 0:
        resp = HttpResponse("exception occurs.")
        resp.headers['Content-Type'] = "text/plain; charset=utf-8"
        resp.headers['exception'] = "999003"
        resp.status_code = 400
        return resp
    # column 이름 앞에 테이블 이름 붙어서 나오는 것을 떼내어 준다
    temp_columns = df.columns
    new_columns = []
    for column in temp_columns:
        temp_name = str(column).split(".")
        if len(temp_name) > 1:
            new_columns.append(temp_name[1].upper())
        else:
            new_columns.append(column)
    df.columns = new_columns

    # 원래 컬럼 리스트 -> 들어온 순서대로 reindex 함
    df.columns = map(str.upper, df.columns)
    print(df.columns)

#    if columns is not None and columns != "null" and columns != "":
#        col_mod = columns.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
#        col_list = col_mod.split(",")
#        origin_index = df.columns
#        new_index = pd.Index(col_list)
#        appended_index = new_index.append(origin_index)
#        dropped_index = appended_index.drop_duplicates(keep='first')
#        df = df.reindex(columns=dropped_index)

    if cocd_columns is not None and cocd_columns != "null" and cocd_columns != "":
        columns_mod = cocd_columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        cocd_mod = cocd_code.replace("[", "")
        cocd_mod = cocd_mod.replace("]", "")
        cocd_mod = cocd_mod.replace("'", "")
        cocd_mod = cocd_mod.replace("\"", "")
        cocd_mod = cocd_mod.replace(" ", "")
        cocd_list = cocd_mod.split(",")

        if cocd_mod != "" and columns_mod != "":
            query = ["SELECT COMM_CD, COMM_NM FROM COCD WHERE "]
            old_list = []
            new_list = []
            for i, cocd in enumerate(cocd_list):
                # print(cocd)
                if i == 0:
                    query.extend("LCD='" + cocd + "' ")
                else:
                    query.extend("OR LCD='" + cocd + "' ")
            query_final = "".join(query)
            # print(query_final)
            # res = list(gongt_connection.execute(query_final))
            cur = connection_tibero.cursor()
            cur.execute(query_final)
            res = cur.fetchall()

            # print(res)
            # print(res)
            for i, v in enumerate(res):
                # print(type(v))
                old_list.append(v[0])
                new_list.append(v[1])
            for column in columns_list:
                column_index = int(df.columns.get_indexer([column])[0])
                # print(column_index)
                if column_index != -1:
                    new_column = str(column) + "_KNM"
                    new_value = df[column].replace(old_list, new_list)
                    df.insert(column_index, new_column, new_value)
                    # query_result[new_column] = query_result[column].replace(old_list, new_list)
            cur.close()

    # data 제외필드 제외하는 부분
    if exclusion is not None:
        exclusion_mod = exclusion.replace("[", "")
        exclusion_mod = exclusion_mod.replace("]", "")
        exclusion_mod = exclusion_mod.replace("'", "")
        exclusion_mod = exclusion_mod.replace("\"", "")
        exclusion_mod = exclusion_mod.replace(" ", "")
        exclusion_list = exclusion_mod.split(",")
        if exclusion_mod != "":
            col_list = set(df.columns)
            drop_list = []
            for exe in exclusion_list:
                if exe in col_list:
                    drop_list.append(exe)
            # print(drop_list)
            if len(drop_list) != 0:
                df = df.drop(drop_list, axis=1)

    # 가져온 데이터의 형식을 downcast하여 메모리 사용량을 줄인다.
    df_int = df.select_dtypes(include=INT_TYPE).columns
    df_float = df.select_dtypes(include=FLOAT_TYPE).columns
    df[df_float] = df[df_float].astype(str)
    df[df_float] = df[df_float].astype(float)
    df_obj = df.select_dtypes(include=['object']).columns
    df_datetime = df.select_dtypes(include=['datetime64']).columns
    # print(query_result[df_datetime].head(50).to_string())
    # print(df_obj)
    # query_result[df_obj] = query_result[df_obj].fillna('')
    df[df_int] = df[df_int].apply(pd.to_numeric, downcast='integer', errors='coerce')
    df[df_float] = df[df_float].apply(pd.to_numeric, downcast='float', errors='coerce')
    df[df_obj] = df[df_obj].astype('category')
    df[df_datetime] = df[df_datetime].astype(str)

    # 컬럼 앞에 MDS 이름 붙이는 부분
    if name is not None:
        columns = df.columns
        rename_columns = []
        for column in columns:
            rename_columns.append(name + "." + str(column))
        df.columns = rename_columns

    print("hive query time:", time.time()-start)
    uid = set_data_to_redis(df)
    print(df)
    print("================================>>>>>>" + uid)
    return HttpResponse(uid, 200)

def mongodb_query(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        name = body['name']
        columns = body['columns']
        exclusion = body['exclusion']
        db = body['tuser']
        col = body['col']
        query = body['nosqlquery']
        projection = body['projection']
        sort = body['sort']
        # limit = body['limit']
        # sort = body['sort']
        # skip = body['skip']

    else:
        name = request.GET.get("name")
        columns = request.GET.get("columns")
        exclusion = request.GET.get("exclusion")
        db = request.GET.get("tuser")
        col = request.GET.get("col")
        query = request.GET.get("nosqlquery")
        projection = request.GET.get("projection")
        sort = request.GET.get("sort")
        # limit = request.GET.get("limit")
        # sort = request.GET.get("sort")
        # skip = request.GET.get("skip")

    # if limit is None:
    #    limit = 1

    # if sort is None:
    #    sort = [("_id", -1)]

    # if skip is None:
    #    skip = 0

    print("name===========================1" + name)
    #print("columns===========================2" + columns)
    print("db=========================== 7" + db)
    print("col=========================== 7" + col)
    print("query=========================== 7" + query)
    print("projection=========================== 7" + projection)
    print("sort=========================== 7" + sort)
    print(json.loads(query))

    mongo = MongoClient(mongo_host, int(mongo_port))
    database = mongo[db]
    collection = database[col]
    query_result = pd.DataFrame()
    try:
        if query is None or query == '':
            cursor = objectIdDecoder(collection.find({}))
            # .sort(sort).limit(limit))
        else:
            json_cache = {}
            json_cache['$match'] = json.loads(query)
            json_cache['$group'] = json.loads(projection)
            print(json_cache)

            #strquery = '{$match: {"YY": "2018"}, $group: {"_id": "$YY", "STD_CNT_SUM": {"$sum": "$STD_CNT"}}}'

            #print(json.loads(strquery))

            #cursor = objectIdDecoder(collection.aggregate(json.loads(strquery)))

            #cursor = objectIdDecoder(collection.find({'YY':'2018'}))
            cursor = objectIdDecoder(collection.find(json.loads(query)))
            #cursor = objectIdDecoder(collection.find(pql.find(query)))

        #Mongdb query결과를 DataFrame 저장
        query_result = pd.DataFrame(list(cursor))

    except Exception as e:
        print(e)
        resp = HttpResponse("exception occurs")
        resp.headers['exception'] = "999002"
        resp.status_code = 400
        return resp
    print(query_result)

    if query_result is None:
        resp = HttpResponse("exception occurs.")
        resp.headers['Content-Type'] = "text/plain; charset=utf-8"
        resp.headers['exception'] = "999003"
        resp.status_code = 400
        return resp

    # 원래 컬럼 리스트 -> 들어온 순서대로 reindex 함
    query_result.columns = map(str.upper, query_result.columns)
    if columns is not None and columns != "null" and columns != "":
        col_mod = columns.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
        col_list = col_mod.split(",")
        origin_index = query_result.columns
        new_index = pd.Index(col_list)
        appended_index = new_index.append(origin_index)
        dropped_index = appended_index.drop_duplicates(keep='first')
        query_result = query_result.reindex(columns=dropped_index)

    print("data 제외필드 제외하는 부분")
    # data 제외필드 제외하는 부분
    if exclusion is not None:
        exclusion_mod = exclusion.replace("[", "")
        exclusion_mod = exclusion_mod.replace("]", "")
        exclusion_mod = exclusion_mod.replace("'", "")
        exclusion_mod = exclusion_mod.replace("\"", "")
        exclusion_mod = exclusion_mod.replace(" ", "")
        exclusion_list = exclusion_mod.split(",")
        if exclusion_mod != "":
            col_list = set(query_result.columns)
            drop_list = []
            for exe in exclusion_list:
                if exe in col_list:
                    drop_list.append(exe)
            # print(drop_list)
            if len(drop_list) != 0:
                query_result = query_result.drop(drop_list, axis=1)

    print("컬럼 앞에 MDS 이름 붙이는 부분")
    # 컬럼 앞에 MDS 이름 붙이는 부분
    if name is not None:
        columns = query_result.columns
        rename_columns = []
        for column in columns:
            rename_columns.append(name + "." + str(column))
        query_result.columns = rename_columns

    uid = set_data_to_redis(query_result)

    return HttpResponse(uid, 200)

def objectIdDecoder(list):
    results = []
    for document in list:
        document['_id'] = str(document['_id'])
        results.append(document)
    return results

def select_query(request):
    #print(request)
    try:
        # global connection
        # global gongt_connection
        # global ghaksa_connection
        # global hangj_connection
        # global resch_connection
        # global buseol_connection
        # global abeek_connection
        # global lifedu_connection
        # global cary_connection
        # global stats_connection
        # global thaksa_connection
        # r = redis.Redis(connection_pool=r_pool)
        # pool = cx_Oracle.SessionPool("HAKSA", "!#HAKSA*",
        #                              "192.168.102.70:1521/SMISBK", min=2, max=10, increment=1, threaded=True,
        #                              encoding="UTF-8")
        # connection = pool.acquire()
        # gongt_connection = cx_Oracle.connect("GONGT", "!#GONGT*", "192.168.102.70:1521/SMISBK?expire_time=2")

        global connection_tibero
        global connection_hive
        start = time.time()
        query = ""
        name = ""
        t_user = ""
        columns = ""
        cocd_columns = ""
        cocd_code = ""
        exclusion = ""
        t_user = ""

        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            query = body['query']
            name = body['name']
            t_user = body['t_user']
            columns = body['columns']
            cocd_columns = body['cocd_columns']
            cocd_code = body['cocd_code']
            exclusion = body['exclusion']
            t_user = body['tuser']

        else:
            query = request.GET.get("query")
            name = request.GET.get("name")
            t_user = request.GET.get("t_user")
            columns = request.GET.get("columns")
            cocd_columns = request.GET.get("cocd_columns")
            cocd_code = request.GET.get("cocd_code")
            exclusion = request.GET.get("exclusion")
            t_user = request.GET.get("tuser")
        # query = request.args.get("query", default=None, type=str)
        # name = request.args.get("name", default=None, type=str)
        # t_user = request.args.get("tuser", default="HAKSA", type=str)
        # columns = request.args.get("columns", default=None, type=str)
        # cocd_columns = request.args.get("cocdcol", default=None)
        # cocd_code = request.args.get("codes", default=None)
        # exclusion = request.args.get("exclusion", default=None)
        # t_user = t_user.upper()

        print("========================================000")
        print(query)
        col_list = []
        print(t_user)
        # res = connection.execute(query)
        # connection.set_character_set('utf8')
        query_result = pd.DataFrame()
        try:
            if t_user == "UPMS":
                # connection = cx_Oracle.connect("HAKSA", "!#HAKSA*", "192.168.102.70:1521/SMISBK?expire_time=2")
                print("*********1")
                query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, connection_tibero, parse_dates=['Date'], chunksize=10000)])
                print(query_result)
                # query_result = pd.read_sql_query(query, connection)
                # connection.close()
            elif t_user == "HIVE":
                query_result = pd.read_sql_query(query, connection_oracle)
                query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, connection_hive, parse_dates=['Date'], chunksize=10000)])
            # elif t_user == "GHAKSA":
            #     # ghaksa_connection = cx_Oracle.connect("GHAKSA", "!#GHAKSA*", "192.168.102.70:1521/SMISBK?expire_time=2")
            #     # query_result = pd.read_sql_query(query, ghaksa_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, ghaksa_connection, parse_dates=['Date'], chunksize=10000)])
            #     # ghaksa_connection.close()
            # elif t_user == "HANGJ":
            #     # hangj_connection = cx_Oracle.connect("HANGJ", "!#HANGJ*", "192.168.102.70:1521/SMISBK?expire_time=2")
            #     # query_result = pd.read_sql_query(query, hangj_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, hangj_connection, parse_dates=['Date'], chunksize=10000)])
            #     # hangj_connection.close()
            # elif t_user == "RESCH":
            #     # resch_connection = cx_Oracle.connect("RESCH", "!#RESCH*", "192.168.102.70:1521/SMISBK?expire_time=2")
            #     # query_result = pd.read_sql_query(query, resch_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, resch_connection, parse_dates=['Date'], chunksize=10000)])
            #     # resch_connection.close()
            # elif t_user == "BUSEOL":
            #     # buseol_connection = cx_Oracle.connect("BUSEOL", "!#BUSEOL*", "192.168.102.70:1521/SMISBK?expire_time=2")
            #     # query_result = pd.read_sql_query(query, buseol_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, buseol_connection, parse_dates=['Date'], chunksize=10000)])
            #     # buseol_connection.close()
            # elif t_user == "ABEEK":
            #     # abeek_connection = cx_Oracle.connect("ABEEK", "!#ABEEK*", "192.168.102.70:1521/SMISBK?expire_time=2")
            #     # query_result = pd.read_sql_query(query, abeek_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, abeek_connection, parse_dates=['Date'], chunksize=10000)])
            #     # abeek_connection.close()
            # elif t_user == "LIFEDU":
            #     # lifedu_connection = cx_Oracle.connect("LIFEDU", "!#LIFEDU*", "192.168.102.70:1521/SMISBK?expire_time=2")
            #     # query_result = pd.read_sql_query(query, lifedu_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, lifedu_connection, parse_dates=['Date'], chunksize=10000)])
            #     # lifedu_connection.close()
            # elif t_user == "CARY":
            #     # cary_connection = cx_Oracle.connect("CARY", "!#CARY*", "192.168.102.70:1521/SMISBK?expire_time=2")
            #     # query_result = pd.read_sql_query(query, cary_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, cary_connection, parse_dates=['Date'], chunksize=10000)])
            #     # cary_connection.close()
            # elif t_user == "STATS":
            #     # stats_connection = cx_Oracle.connect("STATS", "!#STATS*", "192.168.102.70:1521/SMISBK?expire_time=2")
            #     # query_result = pd.read_sql_query(query, stats_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, stats_connection, parse_dates=['Date'], chunksize=10000)])
            #     # stats_connection.close()
            # elif t_user == "THAKSA_NEW":
            #     # thaksa_connection = cx_Oracle.connect("thaksa_new", "THAKSA", "203.247.194.209:1521/XE?expire_time=2")
            #     # query_result = pd.read_sql_query(query, thaksa_connection)
            #     query_result = pd.concat([chunk for chunk in pd.read_sql_query(query, thaksa_connection, parse_dates=['Date'], chunksize=10000)])
            #     # thaksa_connection.close()
            else:
                resp = HttpResponse("exception occurs.")
                resp.headers['exception'] = "999002"
                resp.status_code = 400
                return resp
        except cx_Oracle.DatabaseError:
            logger.debug("DatabaseError... will be new connection open.")
            print("DatabaseError... will be new connection open.")
            # connection = cx_Oracle.connect("HAKSA", "!#HAKSA*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # connection.callTimeout = 0
            # gongt_connection = cx_Oracle.connect("GONGT", "!#GONGT*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # gongt_connection.callTimeout = 0
            # ghaksa_connection = cx_Oracle.connect("GHAKSA", "!#GHAKSA*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # ghaksa_connection.callTimeout = 0
            # hangj_connection = cx_Oracle.connect("HANGJ", "!#HANGJ*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # hangj_connection.callTimeout = 0
            # resch_connection = cx_Oracle.connect("RESCH", "!#RESCH*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # resch_connection.callTimeout = 0
            # buseol_connection = cx_Oracle.connect("BUSEOL", "!#BUSEOL*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # buseol_connection.callTimeout = 0
            # abeek_connection = cx_Oracle.connect("ABEEK", "!#ABEEK*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # abeek_connection.callTimeout = 0
            # lifedu_connection = cx_Oracle.connect("LIFEDU", "!#LIFEDU*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # lifedu_connection.callTimeout = 0
            # cary_connection = cx_Oracle.connect("CARY", "!#CARY*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # cary_connection.callTimeout = 0
            # stats_connection = cx_Oracle.connect("STATS", "!#STATS*", "192.168.102.70:1521/SMISBK?expire_time=2")
            # stats_connection.callTimeout = 0
            # thaksa_connection = cx_Oracle.connect("thaksa_new", "THAKSA", "203.247.194.209:1521/XE?expire_time=2")
            # thaksa_connection.callTimeout = 0
        except Exception as e:
            logger.debug(e)
            resp = HttpResponse("exception occurs")
            resp.headers['exception'] = "999002"
            resp.status_code = 400
            return resp

        if query_result is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # query_result = pd.DataFrame(res.fetchall())

        print(query_result.dtypes)
        # print(query_result.head(50).to_string())
        print("=========================== 3")
        # 원래 컬럼 리스트 -> 들어온 순서대로 reindex 함
        print(query_result.columns)
        query_result.columns = map(str.upper, query_result.columns)
        if columns is not None and columns != "null" and columns != "":
            col_mod = columns.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
            col_list = col_mod.split(",")
            origin_index = query_result.columns
            new_index = pd.Index(col_list)
            appended_index = new_index.append(origin_index)
            dropped_index = appended_index.drop_duplicates(keep='first')
            query_result = query_result.reindex(columns=dropped_index)
        print("=========================== 2")
        # origin_list = query_result.columns.tolist()
        # col_list_final = []
        # for col in col_list:
        #     if col in origin_list:
        #         col_list_final.append(col)
        #         origin_list.remove(col)
        # col_list_final.extend(origin_list)
        # print(origin_list)
        # print(query_result.columns)
        # print(col_list)
        # print(col_list_final)
        # logger.debug(query_result)
        # print(query_result)

        # 가져온 데이터의 형식을 downcast하여 메모리 사용량을 줄인다.
        df_int = query_result.select_dtypes(include=INT_TYPE).columns
        df_float = query_result.select_dtypes(include=FLOAT_TYPE).columns
        query_result[df_float] = query_result[df_float].astype(str)
        query_result[df_float] = query_result[df_float].astype(float)
        df_obj = query_result.select_dtypes(include=['object']).columns
        df_datetime = query_result.select_dtypes(include=['datetime64']).columns
        # print(query_result[df_datetime].head(50).to_string())
        # print(df_obj)
        # query_result[df_obj] = query_result[df_obj].fillna('')
        print("select_query=========================== 4")
        query_result[df_int] = query_result[df_int].apply(pd.to_numeric, downcast='integer', errors='coerce')
        query_result[df_float] = query_result[df_float].apply(pd.to_numeric, downcast='float', errors='coerce')
        query_result[df_obj] = query_result[df_obj].astype('category')
        query_result[df_datetime] = query_result[df_datetime].astype(str)
        # print(query_result[df_datetime].head(50).to_string())
        # print(query_result[df_obj])
        # logger.debug(query_result.dtypes)
        if cocd_columns is not None and cocd_columns != "null" and cocd_columns != "":
            columns_mod = cocd_columns.replace("[", "")
            columns_mod = columns_mod.replace("]", "")
            columns_mod = columns_mod.replace("'", "")
            columns_mod = columns_mod.replace("\"", "")
            columns_mod = columns_mod.replace(" ", "")
            columns_list = columns_mod.split(",")
            cocd_mod = cocd_code.replace("[", "")
            cocd_mod = cocd_mod.replace("]", "")
            cocd_mod = cocd_mod.replace("'", "")
            cocd_mod = cocd_mod.replace("\"", "")
            cocd_mod = cocd_mod.replace(" ", "")
            cocd_list = cocd_mod.split(",")
            print("=========================== 5")
            if cocd_mod != "" and columns_mod != "":
                query = ["SELECT COMM_CD, COMM_NM FROM COCD WHERE "]
                old_list = []
                new_list = []
                for i, cocd in enumerate(cocd_list):
                    # print(cocd)
                    if i == 0:
                        query.extend("LCD='" + cocd + "' ")
                    else:
                        query.extend("OR LCD='"+cocd+"' ")
                query_final = "".join(query)
                # print(query_final)
                # res = list(gongt_connection.execute(query_final))
                cur = connection_tibero.cursor()
                cur.execute(query_final)
                res = cur.fetchall()

                # print(res)
                # print(res)
                print("select_query=========================== 6")
                for i, v in enumerate(res):
                    # print(type(v))
                    old_list.append(v[0])
                    new_list.append(v[1])
                for column in columns_list:
                    column_index = int(query_result.columns.get_indexer([column])[0])
                    # print(column_index)
                    if column_index != -1:
                        new_column = str(column) + "_KNM"
                        new_value = query_result[column].replace(old_list, new_list)
                        query_result.insert(column_index, new_column, new_value)
                        # query_result[new_column] = query_result[column].replace(old_list, new_list)
                cur.close()

        # print("exclusion")
        # data 제외필드 제외하는 부분
        if exclusion is not None:
            print("select_query=========================== 7")
            exclusion_mod = exclusion.replace("[", "")
            exclusion_mod = exclusion_mod.replace("]", "")
            exclusion_mod = exclusion_mod.replace("'", "")
            exclusion_mod = exclusion_mod.replace("\"", "")
            exclusion_mod = exclusion_mod.replace(" ", "")
            exclusion_list = exclusion_mod.split(",")
            if exclusion_mod != "":
                col_list = set(query_result.columns)
                drop_list = []
                for exe in exclusion_list:
                    if exe in col_list:
                        drop_list.append(exe)
                # print(drop_list)
                if len(drop_list) != 0:
                    query_result = query_result.drop(drop_list, axis=1)

        # 컬럼 앞에 MDS 이름 붙이는 부분
        if name is not None:
            columns = query_result.columns
            rename_columns = []
            for column in columns:
                rename_columns.append(name + "." + column)
            query_result.columns = rename_columns
        # print(query_result.head(100).to_string())
        print("time:", time.time()-start)
        print("=========================== 8")
        uid = set_data_to_redis(query_result)
        print("=========================== 9")

        # gongt_connection.close()
        # result = query_result.to_json(orient="records", force_ascii=False, date_format="iso")
        # uid = get_uuid()
        # r.set(uid, result)
        # del [[query_result]]
        # gc.collect()
        # query_result = pd.DataFrame()
        #return uid, 200
        #return Response(uid, status=200 )
        print("select_query= >uid========================== return" + uid)
        return HttpResponse(uid, 200)
        # return str(result), 200
    except Exception as e:
        print("=========================== 1")
        logger.debug(e)
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp



def user_mds(request):
    try:
        # r = redis.Redis(connection_pool=r_pool)
        key = request.args.get("key", type=str, default=None)
        data = request.args.get("data", type=str, default=None)
        logger.debug("key:", key)
        logger.debug("data:", data)
        df = pd.DataFrame
        if key is not None:
            df = get_data_from_redis(key)
        elif data is not None:
            data_dict = json.loads(data)
            df = pd.DataFrame.from_dict(data_dict)
        logger.debug("df:", df.head())
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        uuid1 = set_data_to_redis(df)
        return str(uuid1), 200

        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        #     result = df.to_json(orient='records', force_ascii=False)
        #     uuid1 = get_uuid()
        #     r.set(uuid1, result)
        #     return str(uuid1), 200
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def file_open(request):
    r = redis.Redis(connection_pool=r_pool)
    f = request.files['file']
    if f and allowed_file(f.filename):
        try:
            # data = pd.read_csv(f, encoding='utf-8')
            extension = f.filename.rsplit('.', 1)[1].lower()
            if extension == 'csv':
                data = pd.read_csv(f, error_bad_lines=False, index_col=False)
            elif extension == 'xlsx':
                data = pd.read_excel(f)
            else:
                return
            data = data.rename(columns=lambda s: s.replace(" ", "_"))

            # 가져온 데이터의 형식을 downcast하여 메모리 사용량을 줄인다.
            df_int = data.select_dtypes(include=['int']).columns
            df_float = data.select_dtypes(include=['float']).columns
            df_obj = data.select_dtypes(include=['object']).columns
            data[df_int] = data[df_int].apply(pd.to_numeric, downcast='integer', errors='coerce')
            data[df_float] = data[df_float].apply(pd.to_numeric, downcast='float', errors='coerce')
            data[df_obj] = data[df_obj].astype('category')
            # print(data)
            uid = set_data_to_redis(data)
            # result = data.to_json(orient='records', force_ascii=False, date_format="iso")
            # uid = get_uuid()
            # r.set(uid, result)
            # del [[data]]
            # gc.collect()
            # data = pd.DataFrame()
            return uid, 200
        except Exception as e:
            print(e)
            resp = HttpResponse("file not correct")
            resp.headers['exception'] = "999004"
            resp.status_code = 400
            return resp
    else:
        resp = HttpResponse("file not included or now allowed file extensions")
        resp.headers['exception'] = "999004"
        resp.status_code = 400
        return resp


def string_to_asterisk(x):
    x = str(x)
    if len(x) <= 1:
        return "*"
    new_x = x[:1]
    for i in range(1, len(x)):
        new_x += "*"
    return new_x


def get_data(request):
    # r = redis.Redis(connection_pool=r_pool)
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        key = body['key']
        head = int(body['head'])
        columns = body['columns']
    else:
        key = request.GET.get("key")
        head = int(request.GET.get("head"))
        columns = request.GET.get("columns")
    print("get_data==>key==>"+key + "head==columns=="+columns)
    try:
        #key = request.args.get("key")
        #head = request.args.get("head", type=int, default=0)
        #columns = request.args.get("columns", default=None)

        columns_list = []
        if columns is not None and columns != "null" and columns != "":
            columns_mod = columns.replace("[", "")
            columns_mod = columns_mod.replace("]", "")
            columns_mod = columns_mod.replace("'", "")
            columns_mod = columns_mod.replace("\"", "")
            columns_mod = columns_mod.replace(" ", "")
            columns_list = columns_mod.split(",")
        print("================================================================/get1")
        df = get_data_from_redis(key)

        #print("99999" + df.head(10).to_string())
        # print(df.to_string())
        # print(df.dtypes)

        print("================================================================/get2")

        if df is None:
            print("df is None==>exception occurs.")
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp

        # print("get_data==============1" + df.columns)
        col_list = set(df.columns)
        for col in col_list:
            col_name = col.split(".")[-1]
            if col_name in columns_list:
                df[col] = df[col].apply(lambda x: string_to_asterisk(x))
        # columns_list = col_list.intersection(columns_list)
        # if len(columns_list) != 0:
        #     for col in columns_list:
        #         df[col] = df[col].apply(lambda x: string_to_asterisk(x))
        print("get_data==============2")
        df_float = df.select_dtypes(include=FLOAT_TYPE).columns
        df[df_float] = df[df_float].astype(str)
        df[df_float] = df[df_float].astype(float)
        print("get_data==============3")

        # df_obj = df.select_dtypes(include=['object']).columns
        # query_result[df_obj] = query_result[df_obj].fillna('')
        # df[df_float] = df[df_float].apply(pd.to_numeric, downcast='float', errors='coerce')
        # df[df_obj] = df[df_obj].astype('category')
        print("get_data==============4")
        print(head)

        if head == 0:
            # return str(df.to_dict('records'))
            print("get_data==============5" + df.to_json(orient='records', force_ascii=False, date_format="iso"))
            return HttpResponse(df.to_json(orient='records', force_ascii=False, date_format="iso"), 200)
        else:
            # return str(df.iloc[0:head].to_dict('records'))

            print("get_data==============6")
            print(df.to_json(orient='records'))
            return HttpResponse(df.iloc[0: head].to_json(orient='records', force_ascii=False, date_format="iso"), 200)

        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        #     col_list = set(df.columns)
        #     columns_list = col_list.intersection(columns_list)
        #     if len(columns_list) != 0:
        #         for col in columns_list:
        #             df[col] = df[col].apply(lambda x: string_to_asterisk(x))
        #     if head == 0:
        #         return df.to_json(orient='records', force_ascii=False), 200
        #     else:
        #         return df.iloc[0: head].to_json(orient='records', force_ascii=False), 200
        # else:
        #     resp = HttpResponse("no data")
        #     resp.headers['exception'] = "no data"
        #     resp.status_code = 400
        #     return resp
    except Exception as e:
        print("get_data==============7")
        print(e)
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp



def get_csv_data(request):
    try:
        key = request.args.get("key")
        df = get_data_from_redis(key)
        output_stream = StringIO()
        df.to_csv(output_stream)
        return output_stream.getvalue()
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp



def get_keys(request):
    r = redis.Redis(connection_pool=r_pool)
    keys = r.keys()
    for i, key in enumerate(keys):
        keys[i] = key.decode('ascii')
    return HttpResponse(str(keys), 200)



def del_one(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        key = body['key']
    else:
        key = request.GET.get("key")
    r = redis.Redis(connection_pool=r_pool)
    #key = request.args.get("key")

    print("==========del_one" + key)
    result = r.delete(key)
    return HttpResponse(str(result), 200)



def del_many(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        keys = body['keys']
    else:
        keys = request.GET.get("keys")
    print("==========del_many" + keys)
    r = redis.Redis(connection_pool=r_pool)
    #keys = request.args.get("keys")
    total = 0
    keys_mod = keys.replace("[", "")
    keys_mod = keys_mod.replace("]", "")
    keys_mod = keys_mod.replace("'", "")
    keys_mod = keys_mod.replace("\"", "")
    keys_mod = keys_mod.replace(" ", "")
    print("==========del_many>keys_mod>" + keys_mod)
    keys_list = keys_mod.split(",")
    print("============del_many>keys_list")
    for key in keys_list:
        print(">>>>>>>>>>>>>>>" + key)
        result = r.delete(key)
        total += int(result)
    return HttpResponse(str(total), 200)

