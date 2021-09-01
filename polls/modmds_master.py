# import gc
import json
import uuid
from io import StringIO, BytesIO
import numpy as np
import pandas as pd
import redis
import logging
import pyarrow as pa
import math

from django.http import HttpResponse
from pandas.api.types import is_integer_dtype, is_float_dtype, is_string_dtype, is_object_dtype, is_categorical_dtype, \
    is_datetime64_dtype
from urllib import parse
from wordcloud import WordCloud, ImageColorGenerator
import time

# from pandas.api.types import CategoricalDtype
# import datetime

# import dask.dataframe as dd
# docker tag : seokjoo/modmds
# version : 2.6.4
# msg : merge column multiple argument modify.
# msg(2.3.7) : groupby method change to lowercase
# msg(2.4.0) : groupby delete column_to_int > sum, mean astype float
# msg(2.4.1) : between categorical error fixed ( order category type )
# msg(2.4.2) : between categorical error fixed ( change str type )
# msg(2.4.3) : fillna bug fixed
# msg(2.4.5) : replace " " -> remove
# msg(2.4.6) : group_rank 추가, rank ascend parameter 추가.
# msg(2.4.7) : Exception message 코드화
# msg(2.4.8) : Add some more Exception message
# msg(2.5.0) : Add multiple group input to group_rank
# msg(2.5.1) : Add calculate formula
# msg(2.5.2) : bug fix group rank
# msg(2.5.3) : bug fix calc
# msg(2.5.4) : add column to calc
# msg(2.5.5) : round bug fix
# msg(2.5.6) : wordcloud bug fix
# msg(2.5.7) : round fillna add
# msg(2.5.9) : bug fix round
# msg(2.6.0) : bug fix found read only problem
# msg(2.6.2) : bug fix
# msg(2.6.3) : bug fix
# msg(2.6.4) : bug fix

# redis address ( test and deploy )
# r = redis.Redis(host='203.247.194.215', port=61379)


r_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0)
r = redis.Redis(connection_pool=r_pool)
#r = redis.Redis(host='10.111.82.182', port=6379)

# create logger
logger = logging.getLogger('mod')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


INT_TYPE = {"int_", "int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
FLOAT_TYPE = {"float_", "float", "float16", "float32", "float64"}


def get_data_from_redis(key):
    print("=======================>" + key)
    #print(type(key))
    #print("r.get(key)====>" + r.get(key))
    try:
        # context = pa.default_serialization_context()

        data = pa.deserialize(r.get(key))
        df = pd.DataFrame(data)
        print("-------------------------------")
        return df
    except Exception as e:
        print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        logger.debug(e)
        logger.debug("redis key has no data or not exist.")
        return None
    # except Exception as e:
    #     print(e)
    #     data = r.get(key)
    #     df = pd.DataFrame(json.loads(data))
    #     return df


def set_data_to_redis(df):
    # context = pa.default_serialization_context()
    uuid1 = get_uuid()
    # r.set(uuid1, context.serialize(df).to_buffer().to_pybytes(), datetime.timedelta(seconds=60 * 60 * 24))
    r.set(uuid1, pa.serialize(df).to_buffer().to_pybytes())
    return uuid1


#@app.route('/')
def hello_world():
    return 'modify mds!'


# free dataframe memory
# def free_df(df: pd.DataFrame):
#     del [[df]]
#     gc.collect()
#     df = pd.DataFrame()


# get uuid
def get_uuid():
    uid = uuid.uuid1()
    uid = str(uid).replace("-", "_")
    return uid


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# return # string list (컬럼 리스트)
# ======================================================================================================================
def get_columns(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
        else:
            key = request.GET.get("key")

        #key = request.args.get("key")
        df1 = get_data_from_redis(key)
        if df1 is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data1 = r.get(key)
        # df1 = pd.DataFrame(json.loads(data1))
        columns = df1.columns
        column = []
        for col in columns:
            column.append(col)
        # del [[df1]]
        # gc.collect()
        # df1 = pd.DataFrame()
        #return str(column), 200
        return HttpResponse(str(column), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # columns : 데이터에서 선택할 컬럼 리스트 ([data1, data2, data3 ... ] 형식)
# return # redis key
# ======================================================================================================================
def select_columns(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # df = pd.DataFrame(json.loads(data))
        columns_list_final = []
        col_list = set(df.columns)
        intersection_list = col_list.intersection(columns_list)
        for col in columns_list:
            if col in intersection_list:
                if col not in columns_list_final:
                    columns_list_final.append(col)
        if len(columns_list_final) == 0:
            resp = HttpResponse("Not correct columns")
            resp.headers['exception'] = "999007"
            resp.status_code = 400
            return resp
        df2 = df[[*columns_list_final]]
        uuid1 = set_data_to_redis(df2)
        # del [[df, df2]]
        # gc.collect()
        # df = pd.DataFrame()
        # df2 = pd.DataFrame()
        # result = df2.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # columns : 데이터에서 선택할 컬럼 리스트 ([data1, data2, data3 ... ] 형식)
# param # axis : 0 - row, 1 - column
# return # redis key
# ======================================================================================================================
def drop(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            axis = body['axis']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            axis = request.GET.get("axis")

        #axis = int(request.args.get("axis", default="1"))
        columns_list = []
        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if columns is not None or columns != "null" or columns != "":
            columns_mod = columns.replace("[", "")
            columns_mod = columns_mod.replace("]", "")
            columns_mod = columns_mod.replace("'", "")
            columns_mod = columns_mod.replace("\"", "")
            # columns_mod = columns_mod.replace(" ", "")
            columns_list = columns_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        col_list = set(df.columns)
        intersection_list = col_list.intersection(columns_list)
        columns_list_final = []
        for col in columns_list:
            if col in intersection_list:
                if col not in columns_list_final:
                    columns_list_final.append(col)
        if len(columns_list_final) == 0:
            resp = HttpResponse("Not correct columns")
            resp.headers['exception'] = "999007"
            resp.status_code = 400
            return resp
        try:
            df2 = df.drop(columns_list_final, axis=axis)
        except (KeyError, ValueError) as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        uuid1 = set_data_to_redis(df2)
        # del [[df, df2]]
        # gc.collect()
        # df = pd.DataFrame()
        # df2 = pd.DataFrame()
        # result = df2.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : 데이터에서 이름을 변경할 컬럼
# param # chg-column : 변경할 이름
# return # redis key
# ======================================================================================================================
def change_column(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
            chg_column = body['chg-column']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")
            chg_column = request.GET.get("chg-column")

        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if chg_column is None or chg_column == "" or chg_column == "null":
            resp = HttpResponse("chg_column need one or more parameter")
            resp.headers['exception'] = "999008"
            resp.status_code = 400
            return resp
        columns_mod = column.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        chg_column_mod = chg_column.replace("[", "")
        chg_column_mod = chg_column_mod.replace("]", "")
        chg_column_mod = chg_column_mod.replace("'", "")
        chg_column_mod = chg_column_mod.replace("\"", "")
        # chg_column_mod = chg_column_mod.replace(" ", "")
        chg_column_mod_list = chg_column_mod.split(",")
        if len(columns_list) != len(chg_column_mod_list):
            resp = HttpResponse("column and chg_column must have same parameter length")
            resp.headers['exception'] = "999009"
            resp.status_code = 400
            return resp
        f = dict()
        # object 나 str 타입의 컬럽은 int형 타입으로 형변환 해줌. 안하면 산술 계산시 예상과 다른 값 나옴.
        for i, column in enumerate(columns_list):
            # make dictionary
            f[column] = chg_column_mod_list[i]
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        # print(f)
        try:
            df2 = df.rename(columns=f)
        except Exception as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        # print(df2.to_string())
        uuid1 = set_data_to_redis(df2)
        # del [[df, df2]]
        # gc.collect()
        # df = pd.DataFrame()
        # df2 = pd.DataFrame()
        # result = df2.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
        # else:
        #     resp = Response("999003")
        #     resp.headers['exception'] = "999003"
        #     resp.status_code = 400
        #     return resp
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # columns : 데이터에서 정렬할 컬럼
# param # method : ascend - 오름차순, descend - 내림차순
# return # redis key
# ======================================================================================================================
def sort(request):
    try:

        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            method = body['method']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            method = request.GET.get("method")

        #method = request.args.get("method", default='descend')
        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if method is None or method == "" or method == "null":
            resp = HttpResponse("method need one or more parameter")
            resp.headers['exception'] = "999010"
            resp.status_code = 400
            return resp
        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        method_mod = method.replace("[", "")
        method_mod = method_mod.replace("]", "")
        method_mod = method_mod.replace("'", "")
        method_mod = method_mod.replace("\"", "")
        # method_mod = method_mod.replace(" ", "")
        method_list = method_mod.split(",")
        method_array = ['ascend', 'descend']
        method_arr = []
        for method in method_list:
            if method not in method_array:
                resp = HttpResponse("method out of range.")
                resp.headers['exception'] = "999023"
                resp.status_code = 400
                return resp
            if method == "ascend":
                method_arr.append(True)
            else:
                method_arr.append(False)
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        try:
            df2 = df.sort_values(columns_list, ascending=method_arr, na_position='last')
            # if method == 'ascend':
            #     df2 = df.sort_values(by=[column], axis=0, na_position='last')
            # else:
            #     df2 = df.sort_values(by=[column], axis=0, na_position='last', ascending=False)
        except Exception as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        uuid1 = set_data_to_redis(df2)
        return HttpResponse(str(uuid1), 200)
        #return str(uuid1), 200
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : 기준 컬럼
# param # dfrom : from
# param # dto : to
# return # redis key
# ** 숫자, 문자, 날짜 포맷 가능.
# ** YYYYMMDD 포맷의 날짜 포맷의 경우 YYYY만 넘겨줘도 가능.
# ======================================================================================================================
def between(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            dfrom = body['dfrom']
            dto = body['dto']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            dfrom = request.GET.get("dfrom")
            dto = request.GET.get("dto")

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if dfrom is None or dfrom == "" or dfrom == "null":
            resp = HttpResponse("dfrom need one or more parameter")
            resp.headers['exception'] = "999024"
            resp.status_code = 400
            return resp
        if dto is None or dto == "" or dto == "null":
            resp = HttpResponse("dto need one or more parameter")
            resp.headers['exception'] = "999025"
            resp.status_code = 400
            return resp

        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")

        dfrom_mod = dfrom.replace("[", "")
        dfrom_mod = dfrom_mod.replace("]", "")
        dfrom_mod = dfrom_mod.replace("'", "")
        dfrom_mod = dfrom_mod.replace("\"", "")
        # dfrom_mod = dfrom_mod.replace(" ", "")
        dfrom_list = dfrom_mod.split(",")

        dto_mod = dto.replace("[", "")
        dto_mod = dto_mod.replace("]", "")
        dto_mod = dto_mod.replace("'", "")
        dto_mod = dto_mod.replace("\"", "")
        # dto_mod = dto_mod.replace(" ", "")
        dto_list = dto_mod.split(",")

        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        try:
            if len(columns_list) != len(dfrom_list) or len(columns_list) != len(dto_list):
                resp = HttpResponse("Not correct columns or not same length with columns_list and dfrom, dto")
                resp.headers['exception'] = "999026"
                resp.status_code = 400
                return resp
            for i, column in enumerate(columns_list):
                # print(df[column].dtypes)
                # print(FLOAT_TYPE)
                if str(df[column].dtypes) in FLOAT_TYPE:
                    dfrom_list[i] = float(dfrom_list[i])
                    dto_list[i] = float(dto_list[i])
                elif str(df[column].dtypes) in INT_TYPE:
                    dfrom_list[i] = int(dfrom_list[i])
                    dto_list[i] = int(dto_list[i])
                else:
                    # df[column] = pd.Categorical(df[column], ordered=True)
                    print("order categorical data")
                    print(df[column])
                    df[column] = df[column].astype(str)
                    # df[column] = pd.Categorical(df[column], ordered=True)
                    # df[column] = df[column].sort_values()
                # print(dfrom_list)
                df = df[df[column].between(dfrom_list[i], dto_list[i])]
        except Exception as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        uuid1 = set_data_to_redis(df)
        # result = df.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        return HttpResponse(str(uuid1), 200)
        #return str(uuid1), 200
        # else:
        #     resp = Response("999003")
        #     resp.headers['exception'] = "999003"
        #     resp.status_code = 400
        #     return resp
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis key
# param # columns : 데이터에서 선택할 컬럼 리스트 ([data1, data2, data3 ... ] 형식)
# return # redis key
# 선택한 컬럼을 절대값으로 (음수일때 양수로 변환)
# ======================================================================================================================
def abs_columns(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")


        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        # col_list = set(df.columns)
        # intersection_list = col_list.intersection(columns_list)
        # for col in columns_list:
        #     if col not in intersection_list:
        #         columns_list.remove(col)
        if len(columns_list) == 0:
            resp = HttpResponse("Not correct columns")
            resp.headers['exception'] = "999007"
            resp.status_code = 400
            return resp
        for column in columns_list:
            try:
                df[column] = df[column].astype(str)
                df[column] = df[column].fillna("0")
                # df[column] = df[column].astype('category')
                # df[column] = df[column].fillna("0")
                df = df.astype({column: int})
                df[column] = df[column].abs()
            except Exception as e:
                resp = HttpResponse("exception occurs.")
                resp.headers['exception'] = str(e)
                resp.status_code = 400
                return resp
        uuid1 = set_data_to_redis(df)
        # del [[df]]
        # gc.collect()
        # df = pd.DataFrame()
        # result = df.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        return HttpResponse(str(uuid1), 200)
        #return str(uuid1), 200
        # else:
        #     resp = Response("999003")
        #     resp.headers['exception'] = "999003"
        #     resp.status_code = 400
        #     return resp
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis key
# param # column : column name for count
# return # count number ( string type )
# ======================================================================================================================
def count_column(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")


        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp

        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        try:
            result = df[column].count()
        except Exception as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        return HttpResponse(str(result), 200)
        #return str(result), 200
        # else:
        #     resp = Response("999003")
        #     resp.headers['exception'] = "999003"
        #     resp.status_code = 400
        #     return resp
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis key
# param # column : column name for count
# return # count number ( string type )
# 컬럼에 대한 summary ( int, string 에 따라 결과가 다르다 )
# ======================================================================================================================
def describe_column(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")

        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        try:
            result = df[column].describe()
        except Exception as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        return HttpResponse(str(result.to_json()), 200)
        #return str(result.to_json()), 200
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis key
# param # column : column name for count
# return # count number ( string type )
# 컬럼의 각 값에 대한 count
# ======================================================================================================================
def value_count(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")

        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        try:
            result = df[column].value_counts().to_json()
        except Exception as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        #return str(result), 200
        return HttpResponse(str(result), 200)

    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # query : 가져올 데이터 쿼리
# 1) 비교 연산자 (==, >, >=, <, <=, !=) -> "column == 10"
# 2) in -> "column in [21, 22]" 해당 컬럼의 값이 21이나 22 하나만 해당해도 참
# 3) 논리 연산자 (and, or, not)*모두 소문자 -> "(column1 == 0) and (column2 >= 20)"
# 4) 문자열 부분검색 (str.contains, str.startswith, str.endswith) -> "column.str.contains('졸업')" column 값에 졸업이 들어간 항
# return # uuid
# ======================================================================================================================
#@app.route("/query", methods=['POST'])
def get_data_query(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            query = body['query']
        else:
            key = request.GET.get("key")
            query = request.GET.get("query")


        if query is None or query == "" or query == "null":
            resp = HttpResponse("query need one or more parameter")
            resp.headers['exception'] = "999027"
            resp.status_code = 400
            return resp
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        try:
            result_df = df.query(query)
        except (KeyError, ValueError) as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        uuid1 = set_data_to_redis(result_df)
        # del [[df, result_df]]
        # gc.collect()
        # df = pd.DataFrame()
        # result_df = pd.DataFrame()

        # result = result_df.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
        # else:
        #     resp = Response("999003")
        #     resp.headers['exception'] = "999003"
        #     resp.status_code = 400
        #     return resp
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # columns : or 연산할 컬럼들
# param # methods : 연산할 컬럼들의 방법 (==(eq), !=(neq), >(gt), >=(gte), <(lt), <=(lte), in)
# param # values : 연산할 컬럼들의 비교대상 값(in이 올때는 복수 값이 올수 있으므로 꼭 []로 값을 싸줘야 한다.
# return # uuid
# ======================================================================================================================
def get_data_or(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            methods = body['methods']
            values = body['values']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            methods = request.GET.get("methods")
            values = request.GET.get("values")

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if methods is None or methods == "" or methods == "null":
            resp = HttpResponse("methods need one or more parameter")
            resp.headers['exception'] = "999010"
            resp.status_code = 400
            return resp
        if values is None or values == "" or values == "null":
            resp = HttpResponse("values need one or more parameter")
            resp.headers['exception'] = "999028"
            resp.status_code = 400
            return resp
        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        method_mod = methods.replace("[", "")
        method_mod = method_mod.replace("]", "")
        method_mod = method_mod.replace("'", "")
        method_mod = method_mod.replace("\"", "")
        # method_mod = method_mod.replace(" ", "")
        method_list = method_mod.split(",")
        values_mod = values.replace("\"", "")
        values_mod = values_mod.replace("'", "")
        # values_mod = values_mod.replace(" ", "")
        values_list = values_mod.split(",")

        # data = r.get(key)
        # in 이 여러개 들어올 상황을 대비하여 만든 flag
        in_method_flag = 0
        # in 을 사용했을 때 values 리스트를 찾기 위한 flag
        values_flag = 0
        # query string
        query_string = []
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        # print(df.dtypes)
        for i, column in enumerate(columns_list):
            # 처음 컬럼 처리 부
            if i == 0:
                if method_list[i] == "in":
                    invalue = str(values).split("[")[1 + in_method_flag].split("]")[0].split(",")
                    if str(df[column].dtypes) in FLOAT_TYPE:
                        float_value = []
                        for j, v in enumerate(invalue):
                            float_value.append(float(invalue[j]))
                        query_string.append(f'{column} in {float_value}')
                    elif str(df[column].dtypes) in INT_TYPE:
                        int_value = []
                        for j, v in enumerate(invalue):
                            int_value.append(int(invalue[j]))
                        query_string.append(f'{column} in {int_value}')
                    else:
                        query_string.append(f'{column} in {invalue}')
                    # query_string.append(f'{column} in {invalue}')
                    # print(query_string)
                    in_method_flag += 1
                    values_flag += (len(invalue) - 1)
                else:
                    if method_list[i] == "eq":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f'{column} == {values_list[i]}')
                        else:
                            # string 일 때, " 를 붙여준다.
                            query_string.append(f'{column} == "{values_list[i]}"')
                    elif method_list[i] == "neq":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f'{column} != {values_list[i]}')
                        else:
                            query_string.append(f'{column} != "{values_list[i]}"')
                    elif method_list[i] == "gt":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f'{column} > {values_list[i]}')
                        else:
                            query_string.append(f'{column} > "{values_list[i]}"')
                    elif method_list[i] == "gte":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f'{column} >= {values_list[i]}')
                        else:
                            query_string.append(f'{column} >= "{values_list[i]}"')
                    elif method_list[i] == "lt":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f'{column} < {values_list[i]}')
                        else:
                            query_string.append(f'{column} < "{values_list[i]}"')
                    elif method_list[i] == "lte":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f'{column} <= {values_list[i]}')
                        else:
                            query_string.append(f'{column} <= "{values_list[i]}"')
            # 두번째 이상 컬럼 처리 ( query_string 에 or 붙여서 만들어 마지막에 처리함 )
            else:
                if method_list[i] == "in":
                    invalue = str(values).split("[")[1 + in_method_flag].split("]")[0].split(",")
                    if str(df[column].dtypes) in FLOAT_TYPE:
                        float_value = []
                        for j, v in enumerate(invalue):
                            float_value.append(float(invalue[j]))
                        query_string.append(f' or {column} in {float_value}')
                    elif str(df[column].dtypes) in INT_TYPE:
                        int_value = []
                        for j, v in enumerate(invalue):
                            int_value.append(int(invalue[j]))
                        query_string.append(f' or {column} in {int_value}')
                    else:
                        query_string.append(f' or {column} in {invalue}')
                    # query_string.append(f' or {column} in {invalue}')
                    # print(query_string)
                    # print(invalue)
                    in_method_flag += 1
                    values_flag += (len(invalue) - 1)
                else:
                    if method_list[i] == "eq":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f' or {column} == {values_list[i + values_flag]}')
                        else:
                            query_string.append(f' or {column} == "{values_list[i + values_flag]}"')
                    elif method_list[i] == "neq":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f' or {column} != {values_list[i + values_flag]}')
                        else:
                            query_string.append(f' or {column} != "{values_list[i + values_flag]}"')
                    elif method_list[i] == "gt":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f' or {column} > {values_list[i + values_flag]}')
                        else:
                            query_string.append(f' or {column} > "{values_list[i + values_flag]}"')
                    elif method_list[i] == "gte":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f' or {column} >= {values_list[i + values_flag]}')
                        else:
                            query_string.append(f' or {column} >= "{values_list[i + values_flag]}"')
                    elif method_list[i] == "lt":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f' or {column} < {values_list[i + values_flag]}')
                        else:
                            query_string.append(f' or {column} < "{values_list[i + values_flag]}"')
                    elif method_list[i] == "lte":
                        if str(df[column].dtypes) in INT_TYPE or str(df[column].dtypes) in FLOAT_TYPE:
                            query_string.append(f' or {column} <= {values_list[i + values_flag]}')
                        else:
                            query_string.append(f' or {column} <= "{values_list[i + values_flag]}"')
        query_string = ''.join(query_string)
        logger.debug(query_string)
        try:
            result_df = df.query(query_string)
        except Exception as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        uuid1 = set_data_to_redis(result_df)
        # del [[df, result_df]]
        # gc.collect()
        # df = pd.DataFrame()
        # result_df = pd.DataFrame()
        # result = result_df.to_json(orient='records', force_ascii=False)
        # # print(result)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
        # else:
        #     resp = Response("999003")
        #     resp.headers['exception'] = "999003"
        #     resp.status_code = 400
        #     return resp
    except Exception as e:
        print(e)
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : 컬럼의 이름
# param # newcolumn : 만약 연산한 값들을 새로운 컬럼으로 만드려면 입력 입력 안하면 기존 컬럼에 연산결과를 덮어 씌운다.
# param # slicefrom : 문자열을 자를 시작점
# param # sliceto : 문자열을 자를 마지막점
# param # slicestep : 슬라이스 할 스텝 ( default = 1 )
# 해당 컬럼에서 패턴 문자열을 포함한 row만 가져와서 저장하고 해당 redis key를 반환한다.
# return # uuid ( redis key )
# ======================================================================================================================
def str_slice(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
            new_column = body['newcolumn']
            slice_from = body['slicefrom']
            slice_to = body['sliceto']
            slice_step = body['slicestep']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")
            new_column = request.GET.get("newcolumn")
            slice_from = request.GET.get("slicefrom")
            slice_to = request.GET.get("sliceto")
            slice_step = request.GET.get("slicestep")

        #slice_step = request.args.get("slicestep", default=1, type=int)

        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp
        if slice_from is None or slice_from == "" or slice_from == "null":
            resp = HttpResponse("slice_from need one or more parameter")
            resp.headers['exception'] = "999029"
            resp.status_code = 400
            return resp
        if column is None or column == "" or column == "null":
            resp = HttpResponse("slice_to need one or more parameter")
            resp.headers['exception'] = "999030"
            resp.status_code = 400
            return resp

        # new column이 빈 값이면 기존 column을 가공
        if new_column == "" or new_column is None or new_column == "null":
            new_column = column
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        df[column] = df[column].astype(str)
        df[column] = df[column].fillna("")
        df[column] = df[column].astype(str)
        if slice_from == "null":
            slice_to_int = int(slice_to)
            df[new_column] = df[column].str.slice(stop=slice_to_int, step=slice_step)
        elif slice_to == "null":
            slice_from_int = int(slice_from)
            if slice_from_int < 0:
                df[new_column] = df[column].str.slice(start=slice_from_int, step=slice_step)
            else:
                df[new_column] = df[column].str.slice(start=slice_from_int - 1, step=slice_step)
        else:
            slice_to_int = int(slice_to)
            slice_from_int = int(slice_from)
            if slice_from_int < 0:
                df[new_column] = df[column].str.slice(slice_from_int, slice_to_int, slice_step)
            else:
                df[new_column] = df[column].str.slice(slice_from_int - 1, slice_to_int, slice_step)
        uuid1 = set_data_to_redis(df)
        # del [[df]]
        # gc.collect()
        # df = pd.DataFrame()
        # result = df.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
        # else:
        #     resp = Response("999003")
        #     resp.headers['exception'] = "999003"
        #     resp.status_code = 400
        #     return resp
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : 컬럼의 이름
# param # newcolumn : 만약 연산한 값들을 새로운 컬럼으로 만드려면 입력 입력 안하면 기존 컬럼에 연산결과를 덮어 씌운다.
# param # conditions : 조건 ( one or array )
# param # values : 변경할 값 ( one or array )
# conditions 와 values 의 길이가 같아야 함. (,로 구분)
# ======================================================================================================================
def decode(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
            new_column = body['newcolumn']
            conditions = body['conditions']
            values = body['values']
            default_value = body['default']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")
            new_column = request.GET.get("newcolumn")
            conditions = request.GET.get("conditions")
            values = request.GET.get("values")
            default_value = request.GET.get("default")

        #new_column = request.args.get("newcolumn", default="")
        #default_value = request.args.get("default", default=None)
        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp
        if conditions is None or conditions == "" or conditions == "null":
            resp = HttpResponse("conditions need one or more parameter")
            resp.headers['exception'] = "999031"
            resp.status_code = 400
            return resp
        if values is None or values == "" or values == "null":
            resp = HttpResponse("values need one or more parameter")
            resp.headers['exception'] = "999032"
            resp.status_code = 400
            return resp

        conditions = conditions.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
        condition_list = conditions.split(",")
        values = values.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
        value_list = values.split(",")
        if new_column == "" or new_column is None or new_column == "null":
            new_column = column
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        df[column] = df[column].astype(str)
        df[column] = df[column].fillna("")
        df[column] = df[column].astype(str)
        # for i, condition in enumerate(condition_list):
        if default_value is not None and default_value != "nil" and default_value != "null" and default_value != "":
            df[new_column] = df[column].mask(df[column].isin(condition_list) == False, default_value)
        else:
            df[new_column] = df[column]
        df[new_column] = df[new_column].replace(condition_list, value_list)
        uuid1 = set_data_to_redis(df)
        # del [[df]]
        # gc.collect()
        # df = pd.DataFrame()
        # result = df.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : 비교할 컬럼의 이름
# param # pattern : 패턴 문자열을 포함한 컬럼을 찾기위한 문자열
# 해당 컬럼에서 패턴 문자열을 포함한 row만 가져와서 저장하고 해당 redis key를 반환한다.
# return # uuid ( redis key )
# ======================================================================================================================
def contain(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            patterns = body['patterns']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            patterns = request.GET.get("patterns")

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if patterns is None or patterns == "" or patterns == "null":
            resp = HttpResponse("patterns need one or more parameter")
            resp.headers['exception'] = "999033"
            resp.status_code = 400
            return resp

        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        patterns_mod = patterns.replace("[", "")
        patterns_mod = patterns_mod.replace("]", "")
        patterns_mod = patterns_mod.replace("'", "")
        patterns_mod = patterns_mod.replace("\"", "")
        # patterns_mod = patterns_mod.replace(" ", "")
        patterns_list = patterns_mod.split(",")
        if len(columns_list) != len(patterns_list):
            resp = HttpResponse("Not correct columns or not same length with columns_list and patterns_list")
            resp.headers['exception'] = "999034"
            resp.status_code = 400
            return resp
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        for i, column in enumerate(columns_list):
            df = df[df[column].str.contains(patterns_list[i])]
        uuid1 = set_data_to_redis(df)
        # del [[df]]
        # gc.collect()
        # df = pd.DataFrame()
        # result = df.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : 비교할 컬럼의 이름
# param # method : 숫자를 비교할 방법 eq ==, neq !=, gt >, gte >=, lt <, lte <=
# param # value : 비교할 숫자
# param # intorfloat : 비교할 컬럼을 int 형으로 변환 할건지 float으로 변환할건지, 빈값은 0 혹은 0.0으로 채워짐
# 해당 컬럼에서 패턴 문자열을 포함한 row만 가져와서 저장하고 해당 redis key를 반환한다.
# return # uuid ( redis key )
# ======================================================================================================================
def comparison(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
            method = body['method']
            value = body['value']
            intorfloat = body['intorfloat']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")
            method = request.GET.get("method")
            value = request.GET.get("value")
            intorfloat = request.GET.get("intorfloat")

        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp
        if method is None or method == "" or method == "null":
            resp = HttpResponse("method need one or more parameter")
            resp.headers['exception'] = "999011"
            resp.status_code = 400
            return resp
        if value is None or value == "" or value == "null":
            resp = HttpResponse("value need one or more parameter")
            resp.headers['exception'] = "999035"
            resp.status_code = 400
            return resp
        if intorfloat is None or intorfloat == "" or intorfloat == "null":
            resp = HttpResponse("intorfloat need one or more parameter")
            resp.headers['exception'] = "999036"
            resp.status_code = 400
            return resp
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))

        # column values change to int or float type
        if intorfloat == "int":
            df[column] = df[column].astype(str)
            df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
            df[column] = df[column].fillna("0")
            df = df.astype({column: int})
            value = int(value)
        elif intorfloat == "float":
            df[column] = df[column].astype(str)
            df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
            df[column] = df[column].fillna("0.0")
            df = df.astype({column: float})
            value = float(value)
        else:
            resp = HttpResponse("intorfloat out of range.")
            resp.headers['exception'] = "999037"
            resp.status_code = 400
            return resp

        if method == 'eq':
            df = df[df[column] == value]
        elif method == 'neq':
            df = df[df[column] != value]
        elif method == 'gt':
            df = df[df[column] > value]
        elif method == 'gte':
            df = df[df[column] >= value]
        elif method == 'lt':
            df = df[df[column] < value]
        elif method == 'lte':
            df = df[df[column] <= value]
        else:
            resp = HttpResponse("method out of range.")
            resp.headers['exception'] = "999038"
            resp.status_code = 400
            return resp
        uuid1 = set_data_to_redis(df)
        # del [[df]]
        # gc.collect()
        # df = pd.DataFrame()
        # result = df.to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : 비교할 컬럼의 이름
# param # value : 문자열이 들어간 열만 갖고 오기 위한 값 복수개이면 쉽표로 구분 -> a, b, c
# 해당 컬럼에서 value값을 갖고 있는 열만 가져와서 저장하고 해당 redis key를 반환한다.
# return # uuid ( redis key )
# ======================================================================================================================
def str_in(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
            value = body['value']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")
            value = request.GET.get("value")

        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp
        if value is None or value == "" or value == "null":
            resp = HttpResponse("value need one or more parameter")
            resp.headers['exception'] = "999033"
            resp.status_code = 400
            return resp

        value_mod = value.replace("[", "")
        value_mod = value_mod.replace("]", "")
        value_mod = value_mod.replace("'", "")
        value_mod = value_mod.replace("\"", "")
        # value_mod = value_mod.replace(" ", "")
        value_list = value_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        df = df[df[column].isin(value_list)]
        # print(df)
        uuid1 = set_data_to_redis(df)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : rank 적용할 컬럼 이름
# param # method : rank method []
# return # uuid
# ======================================================================================================================
def get_data_rank(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
            ascend = body['ascend']
            method = body['method']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")
            ascend = request.GET.get("ascend")
            method = request.GET.get("method")


        #ascend = request.args.get("ascend", default="ascend")
        #method = request.args.get("method", default='min')
        if ascend == "ascend":
            ascend_final = True
        elif ascend == "descend":
            ascend_final = False
        else:
            ascend_final = True
        if column is None or column == "" or column == "null":
            resp = HttpResponse("column need one or more parameter")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp

        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        # if str(df[column].dtypes) not in FLOAT_TYPE or str(df[column].dtypes) not in INT_TYPE:
        #     df[column] = df[column].astype(float)
        df['rank'] = df[column].rank(method=method, ascending=ascend_final)
        uuid1 = set_data_to_redis(df)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : rank 적용할 컬럼 이름
# param # method : rank method []
# return # uuid
# ======================================================================================================================
def group_rank(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            column = body['column']
            group = body['group']
            ascend = body['ascend']
            method = body['method']
        else:
            key = request.GET.get("key")
            column = request.GET.get("column")
            group = request.GET.get("group")
            ascend = request.GET.get("ascend")
            method = request.GET.get("method")


        #ascend = request.args.get("ascend", default="ascend")
        #method = request.args.get("method", default='min')
        group_list = []
        if ascend == "ascend":
            ascend_final = True
        elif ascend == "descend":
            ascend_final = False
        else:
            ascend_final = True

        if column is None or column == "" or column == "null":
            resp = HttpResponse("999005")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp
        if group is None or group == "" or group == "null":
            resp = HttpResponse("group need one or more parameter")
            resp.headers['exception'] = "999022"
            resp.status_code = 400
            return resp
        else:
            group_mod = group.replace("[", "")
            group_mod = group_mod.replace("]", "")
            group_mod = group_mod.replace("'", "")
            group_mod = group_mod.replace("\"", "")
            # columns_mod = columns_mod.replace(" ", "")
            group_list = group_mod.split(",")

        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        if str(df[column].dtypes) not in FLOAT_TYPE or str(df[column].dtypes) not in INT_TYPE:
            df[column] = df[column].astype(float)
        new_column_name = "rank_" + column
        df[new_column_name] = df.groupby(group_list)[column].rank(method=method, ascending=ascend_final)
        uuid1 = set_data_to_redis(df)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# 로지스틱 threshold 판별 function
def round_calc(value, ndeci):
    if value == 0:
        return 0
    else:
        value = np.round(value * math.pow(10, int(ndeci))) / math.pow(10, int(ndeci))
        print(value)
        # print(np.round(value * math.pow(10, int(ndeci))) / math.pow(10, int(ndeci)))
        return value


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # column : round 적용할 컬럼 이름
# param # ndecimal : 반올림 하여 나타낼 소수점 자리수
# return # uuid
# ======================================================================================================================
def get_data_round(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            ndecimal = body['ndecimal']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            ndecimal = request.GET.get("ndecimal")

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if ndecimal is None or ndecimal == "" or ndecimal == "null":
            resp = HttpResponse("ndecimal need one or more parameter")
            resp.headers['exception'] = "999039"
            resp.status_code = 400
            return resp

        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        # number_of_decimal = int(request.args.get("ndecimal", default="2"))
        ndecimal_mod = ndecimal.replace("[", "")
        ndecimal_mod = ndecimal_mod.replace("]", "")
        ndecimal_mod = ndecimal_mod.replace("'", "")
        ndecimal_mod = ndecimal_mod.replace("\"", "")
        # ndecimal_mod = ndecimal_mod.replace(" ", "")
        ndecimal_list = ndecimal_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        columns_list_final = []
        col_list = set(df.columns)
        intersection_list = col_list.intersection(columns_list)
        for col in columns_list:
            if col in intersection_list:
                if col not in columns_list_final:
                    columns_list_final.append(col)

        if len(columns_list_final) != len(ndecimal_list):
            resp = HttpResponse("Not correct columns or not same length with columns_list and method list")
            resp.headers['exception'] = "999040"
            resp.status_code = 400
            return resp
        df2 = df.copy()
        for i, column in enumerate(columns_list_final):
            # print(column, ndecimal_list[i])
            print(df[column].dtype)
            if str(df[column].dtype) not in FLOAT_TYPE:
                df2[column] = df2[column].astype(float)
                if df[column].isnull().sum() > 1:
                    df2[column] = df[column].fillna(value=0)
                # print(df[column].dtype)
            else:
                if df[column].isnull().sum() > 1:
                    df2[column] = df[column].fillna(value=0)
                # print(df[column].dtype)
            # print(ndecimal_list[i])
            ndeci = int(ndecimal_list[i])
            print(ndeci)
            # print(df.head(100).to_string())
            # print(column, ndeci)
            # df[column] = np.round(df[column], decimals=ndeci)
            # df = df.round({column: int(ndecimal_list[i])})
            # df[column] = df[column].apply(lambda x: np.ceil(x * math.pow(10, int(ndeci))) / math.pow(10, int(ndeci)))
            df2[column] = df2[column].apply(lambda x: float(round_calc(x, ndeci)))
            # print(df2[column])
            df2[column] = df2[column].astype(str)
            df2[column] = df2[column].astype(float)
        print(df2.head(100).to_string())
        # print(df.to_string())
        uuid1 = set_data_to_redis(df2)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# aggregate & group by
# param # key : redis 에서 가져올 데이터의 key
# param # group : group by 할 컬럼 (1 or N) N일때는 ','로 구분 (ex - SHREG_CD, JUNG_CD)
# param # columns : 그룹화 하여 나타낼 컬럼 (1 or N) N일때는 ','로 구분 (ex - SHREG_CD, JUNG_CD)
# param # method : 그룹화 하여 보여줄 값의 함수 (sum, count, min, max, mean, std, first, last, quantile, var, median)
# param # columntoint : columns 를 int로 변환할건지 여부 ( default : false ) sum, mean 같은거 하려면 변환해줘야 함.
# return # uuid
# ======================================================================================================================
def get_data_groupby(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            group = body['group']
            method = body['method']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            group = request.GET.get("group")
            method = request.GET.get("method")

        #key = request.args.get("key")
        #group = request.args.get("group", default=None)
        #columns = request.args.get("columns", default=None)
        #method = request.args.get("method", default=None)

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if group is None or group == "" or group == "null":
            resp = HttpResponse("ndecimal need one or more parameter")
            resp.headers['exception'] = "999012"
            resp.status_code = 400
            return resp
        if method is None or method == "" or method == "null":
            resp = HttpResponse("method need one or more parameter")
            resp.headers['exception'] = "999010"
            resp.status_code = 400
            return resp
        method = method.lower()
        group_list = []
        columns_list = []
        method_list = []
        column_to_int_list = []
        if group is not None and group != "" and group != "null":
            group_mod = group.replace("[", "")
            group_mod = group_mod.replace("]", "")
            group_mod = group_mod.replace("'", "")
            group_mod = group_mod.replace("\"", "")
            # group_mod = group_mod.replace(" ", "")
            group_list = group_mod.split(",")
        if columns is not None and columns != "" and columns != "null":
            columns_mod = columns.replace("[", "")
            columns_mod = columns_mod.replace("]", "")
            columns_mod = columns_mod.replace("'", "")
            columns_mod = columns_mod.replace("\"", "")
            # columns_mod = columns_mod.replace(" ", "")
            columns_list = columns_mod.split(",")
        if method is not None and method != "" and method != "null":
            method_mod = method.replace("[", "")
            method_mod = method_mod.replace("]", "")
            method_mod = method_mod.replace("'", "")
            method_mod = method_mod.replace("\"", "")
            # method_mod = method_mod.replace(" ", "")
            method_list = method_mod.split(",")
        # if column_to_int is not None and column_to_int != "" and column_to_int != "null":
        #     column_to_int_mod = column_to_int.replace("[", "")
        #     column_to_int_mod = column_to_int_mod.replace("]", "")
        #     column_to_int_mod = column_to_int_mod.replace("'", "")
        #     column_to_int_mod = column_to_int_mod.replace("\"", "")
        #     column_to_int_mod = column_to_int_mod.replace(" ", "")
        #     column_to_int_list = column_to_int_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp

        f = dict()
        # object 나 str 타입의 컬럽은 int형 타입으로 형변환 해줌. 안하면 산술 계산시 예상과 다른 값 나옴.
        for i, column in enumerate(columns_list):
            # make dictionary
            if column in f:
                f[column].append(method_list[i])
            else:
                f[column] = [method_list[i]]
            # if column_to_int is true make column to int type
            # if column_to_int_list is not None and column_to_int_list[i] == "true":
            #     # print(column_to_int_list[i])
            #     # df[column].replace('None', np.nan, inplace=True)
            #     # df[column].fillna("0")
            #     df[column] = df[column].astype(str)
            #     df[column] = df[column].replace(np.nan, "0")
            #     df = df.astype({column: int})
            # sum, count, min, max, mean, std, first, last, quantile, var, median
            try:
                if method_list[i] == "sum" or method_list[i] == "mean":
                    df[column] = df[column].astype(str)
                    df[column] = df[column].replace(np.nan, "0")
                    df = df.astype({column: float})
            except Exception as e:
                logger.debug(e)
                resp = HttpResponse("exception occurs.")
                resp.headers['exception'] = "999013"
                resp.status_code = 400
                return resp
        # print(f)
        # result_df = df.groupby(group_list)[[*columns_list]]
        # res_df = result_df.agg(method_list)
        # print(result_df.agg(['size', 'mean', 'std', 'min', 'max', 'count']))
        result_df = df.groupby(group_list)
        res_df = result_df.agg(f)
        res_df.columns = ['_'.join(x) if isinstance(x, tuple) else x for x in res_df.columns.ravel()]
        # result = res_df.reset_index().to_json(force_ascii=False)
        uuid1 = set_data_to_redis(res_df.reset_index())
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        print(e)
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# aggregate & group by
# param # key : redis 에서 가져올 데이터의 key
# param # group : group by 할 컬럼 (1 or N) N일때는 ','로 구분 (ex - SHREG_CD, JUNG_CD)
# param # columns : 그룹화 하여 나타낼 컬럼 (1 or N) N일때는 ','로 구분 (ex - SHREG_CD, JUNG_CD)
# param # method : 그룹화 하여 보여줄 값의 함수 (sum, count, min, max, mean, std, first, last, quantile, var, median)
# param # columntoint : columns 를 int로 변환할건지 여부 ( default : false ) sum, mean 같은거 하려면 변환해줘야 함.
# return # uuid
# ======================================================================================================================
def get_data_groupby_condition(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            group = body['group']
            new_column = body['new_columns']
            new_method = body['new_methods']
            target_column = body['target_columns']
            target_method = body['target_methods']
            target_value = body['target_values']
        else:
            key = request.GET.get("key")
            group = request.GET.get("group")
            new_column = request.GET.get("new_columns")
            new_method = request.GET.get("new_methods")
            target_column = request.GET.get("target_columns")
            target_method = request.GET.get("target_methods")
            target_value = request.GET.get("target_values")

        #group = request.args.get("group", default=None)
        #new_column = request.args.get("new_columns", default=None)
        #new_method = request.args.get("new_methods", default=None)
        #target_column = request.args.get("target_columns", default=None)
        #target_method = request.args.get("target_methods", default=None)
        #target_value = request.args.get("target_values", default=None)

        new_method = new_method.lower()
        target_method = target_method.lower()
        # new_column = 'TU_0001_00023.AVG_MAK, TU_0001_00023.STD_NO, TU_0001_00023.AVG_MAK, TU_0001_00023.STD_NO, ' \
        #              'TU_0001_00023.STD_NO'
        # new_method = 'mean, count, mean, count, count'
        # target_column = 'TU_0001_00023.GEN_GBN, TU_0001_00023.GEN_GBN, TU_0001_00023.GEN_GBN,
        # TU_0001_00023.GEN_GBN, ' \
        #                 'TU_0001_00023.AVG_MAK'
        # target_method = 'in, in, in, in, between'
        # target_value = 'F, F, M, M, 0:5'  # int

        group_list = []
        new_column_list = []
        new_method_list = []
        target_column_list = []
        target_method_list = []
        target_value_list = []

        if group is not None and group != "" and group != "null":
            group_mod = group.replace("[", "")
            group_mod = group_mod.replace("]", "")
            group_mod = group_mod.replace("'", "")
            group_mod = group_mod.replace("\"", "")
            # group_mod = group_mod.replace(" ", "")
            group_list = group_mod.split(",")

        # new column name 은 사용자에게 받을 것인지 아니면 new_column + method 이름으로 갈건지 후자가 나을듯
        # if new_column_name is not None and new_column_name != "" and new_column_name != "null":
        #     new_column_name_mod = new_column_name.replace("[", "").replace("]", "") \
        #         .replace("'", "").replace("\"", "").replace(" ", "")
        #     new_column_name_list = new_column_name_mod.split(",")
        if new_column is not None and new_column != "" and new_column != "null":
            new_column_mod = new_column.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            new_column_list = new_column_mod.split(",")
        if new_method is not None and new_method != "" and new_method != "null":
            new_method_mod = new_method.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            new_method_list = new_method_mod.split(",")
        if target_column is not None and target_column != "" and target_column != "null":
            target_column_mod = target_column.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            target_column_list = target_column_mod.split(",")
        if target_method is not None and target_method != "" and target_method != "null":
            target_method_mod = target_method.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            target_method_list = target_method_mod.split(",")
        if target_value is not None and target_value != "" and target_value != "null":
            target_value_mod = target_value.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            target_value_list = target_value_mod.split(",")

        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        result = pd.DataFrame()
        if len(new_column_list) != 0:
            # if len(columns_list) == 0:
            result = df.groupby(group_list).count()
            result = result.iloc[:, 0:0]
            for i, new_col in enumerate(new_column_list):
                new_name = "[" + str(i + 1) + "]" + new_col + "_" + new_method_list[i] + "_" + target_method_list[i] \
                           + "_" + target_value_list[i]
                if target_method_list[i] == "gt":
                    tmp_group = df[df[target_column_list[i]] > int(target_value_list[i])].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "gte":
                    tmp_group = df[df[target_column_list[i]] >= int(target_value_list[i])].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "lt":
                    tmp_group = df[df[target_column_list[i]] < int(target_value_list[i])].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "lte":
                    tmp_group = df[df[target_column_list[i]] <= int(target_value_list[i])].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "eq":
                    tmp_group = df[df[target_column_list[i]] == target_value_list[i]].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "neq":
                    tmp_group = df[df[target_column_list[i]] != target_value_list[i]].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "in":
                    value = target_value_list[i]
                    value_arr = value.split(":")
                    # print(value_arr)
                    tmp_group = df[df[target_column_list[i]].isin(value_arr)].groupby(group_list)
                    # print(tmp_group.agg({new_col: new_method_list[i]}))
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "between":
                    value = target_value_list[i]
                    value_arr = value.split(":")
                    if len(value_arr) != 2:
                        resp = HttpResponse("exception occurs.")
                        resp.headers['exception'] = "999014"
                        resp.status_code = 400
                        return resp
                    tmp_group = df[(df[target_column_list[i]] >= float(value_arr[0])) &
                                   (df[target_column_list[i]] < float(value_arr[1]))].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                else:
                    # print("method error")
                    resp = HttpResponse("exception occurs.")
                    resp.headers['exception'] = "999015"
                    resp.status_code = 400
                    return resp
        result.columns = ['_'.join(x) if isinstance(x, tuple) else x for x in result.columns.ravel()]
        # result = res_df.reset_index().to_json(force_ascii=False)
        uuid1 = set_data_to_redis(result.reset_index())
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# aggregate & group by
# param # key : redis 에서 가져올 데이터의 key
# param # group : group by 할 컬럼 (1 or N) N일때는 ','로 구분 (ex - SHREG_CD, JUNG_CD)
# param # columns : 그룹화 하여 나타낼 컬럼 (1 or N) N일때는 ','로 구분 (ex - SHREG_CD, JUNG_CD)
# param # method : 그룹화 하여 보여줄 값의 함수 (sum, count, min, max, mean, std, first, last, quantile, var, median)
# param # columntoint : columns 를 int로 변환할건지 여부 ( default : false ) sum, mean 같은거 하려면 변환해줘야 함.
# return # uuid
# ======================================================================================================================
def get_data_groupby_condition_both(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            group = body['group']
            method = body['methods']
            new_column = body['new_columns']
            new_method = body['new_methods']
            target_column = body['target_columns']
            target_method = body['target_methods']
            target_value = body['target_values']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            group = request.GET.get("group")
            method = request.GET.get("methods")
            new_column = request.GET.get("new_columns")
            new_method = request.GET.get("new_methods")
            target_column = request.GET.get("target_columns")
            target_method = request.GET.get("target_methods")
            target_value = request.GET.get("target_values")

        #columns = request.args.get("columns", default=None)
        #group = request.args.get("group", default=None)
        #method = request.args.get("methods", default=None)
        #new_column = request.args.get("new_columns", default=None)
        #new_method = request.args.get("new_methods", default=None)
        #target_column = request.args.get("target_columns", default=None)
        #target_method = request.args.get("target_methods", default=None)
        #target_value = request.args.get("target_values", default=None)

        if method is not None:
            method = method.lower()
        if new_method is not None:
            new_method = new_method.lower()
        if target_method is not None:
            target_method = target_method.lower()

        # new_column = 'TU_0001_00023.AVG_MAK, TU_0001_00023.STD_NO, TU_0001_00023.AVG_MAK, TU_0001_00023.STD_NO, ' \
        #              'TU_0001_00023.STD_NO'
        # new_method = 'mean, count, mean, count, count'
        # target_column = 'TU_0001_00023.GEN_GBN, TU_0001_00023.GEN_GBN, TU_0001_00023.GEN_GBN,
        # TU_0001_00023.GEN_GBN, ' \
        #                 'TU_0001_00023.AVG_MAK'
        # target_method = 'in, in, in, in, between'
        # target_value = 'F, F, M, M, 0:5'  # int

        group_list = []
        columns_list = []
        method_list = []
        new_column_list = []
        new_method_list = []
        target_column_list = []
        target_method_list = []
        target_value_list = []

        if group is not None and group != "" and group != "null":
            group_mod = group.replace("[", "")
            group_mod = group_mod.replace("]", "")
            group_mod = group_mod.replace("'", "")
            group_mod = group_mod.replace("\"", "")
            # group_mod = group_mod.replace(" ", "")
            group_list = group_mod.split(",")
        if columns is not None and columns != "" and columns != "null":
            columns_mod = columns.replace("[", "")
            columns_mod = columns_mod.replace("]", "")
            columns_mod = columns_mod.replace("'", "")
            columns_mod = columns_mod.replace("\"", "")
            # columns_mod = columns_mod.replace(" ", "")
            columns_list = columns_mod.split(",")
        if method is not None and method != "" and method != "null":
            method_mod = method.replace("[", "")
            method_mod = method_mod.replace("]", "")
            method_mod = method_mod.replace("'", "")
            method_mod = method_mod.replace("\"", "")
            # method_mod = method_mod.replace(" ", "")
            method_list = method_mod.split(",")

        df = get_data_from_redis(key)

        # 먼저 그룹 바이 하는 부분 ( 조건 없는 groupby )
        # print(columns_list)
        group = df.groupby(group_list)
        if len(columns_list) != 0:
            f = dict()
            for i, column in enumerate(columns_list):
                # make dictionary
                if column in f:
                    f[column].append(method_list[i])
                else:
                    f[column] = [method_list[i]]
            result = group.agg(f)
            # result['3.0초과'] = df.filter(lambda x: x['TU_0001_00023.AVG_MAK'] > 3.0).groupby('TU_0001_00023.MJ_CD')\
            #     .agg({'TU_0001_00023.STD_NO': 'count'})
            # print(df.filter(lambda x: x['TU_0001_00023.AVG_MAK'] > 3.0))
            # print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
            # print(result2)
            result.columns = [['_'.join(x) if isinstance(x, tuple) else x for x in result.columns.ravel()]]
        else:
            result = group.count()
            result = result.iloc[:, 0:0]

        # new column name 은 사용자에게 받을 것인지 아니면 new_column + method 이름으로 갈건지 후자가 나을듯
        # if new_column_name is not None and new_column_name != "" and new_column_name != "null":
        #     new_column_name_mod = new_column_name.replace("[", "").replace("]", "") \
        #         .replace("'", "").replace("\"", "").replace(" ", "")
        #     new_column_name_list = new_column_name_mod.split(",")
        if new_column is not None and new_column != "" and new_column != "null":
            new_column_mod = new_column.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            new_column_list = new_column_mod.split(",")
        if new_method is not None and new_method != "" and new_method != "null":
            new_method_mod = new_method.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            new_method_list = new_method_mod.split(",")
        if target_column is not None and target_column != "" and target_column != "null":
            target_column_mod = target_column.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            target_column_list = target_column_mod.split(",")
        if target_method is not None and target_method != "" and target_method != "null":
            target_method_mod = target_method.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            target_method_list = target_method_mod.split(",")
        if target_value is not None and target_value != "" and target_value != "null":
            target_value_mod = target_value.replace("[", "").replace("]", "") \
                .replace("'", "").replace("\"", "")
            target_value_list = target_value_mod.split(",")

        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # result = pd.DataFrame()
        if len(new_column_list) != 0:
            # if len(columns_list) == 0:
            # result = df.groupby(group_list).count()
            # result = result.iloc[:, 0:0]
            for i, new_col in enumerate(new_column_list):
                new_name = "[" + str(i + 1) + "]" + new_col + "_" + new_method_list[i] + "_" + target_method_list[i] \
                           + "_" + target_value_list[i]
                if target_method_list[i] == "gt":
                    tmp_group = df[df[target_column_list[i]] > int(target_value_list[i])].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "gte":
                    tmp_group = df[df[target_column_list[i]] >= int(target_value_list[i])].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "lt":
                    tmp_group = df[df[target_column_list[i]] < int(target_value_list[i])].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "lte":
                    tmp_group = df[df[target_column_list[i]] <= int(target_value_list[i])].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "eq":
                    tmp_group = df[df[target_column_list[i]] == target_value_list[i]].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "neq":
                    tmp_group = df[df[target_column_list[i]] != target_value_list[i]].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "in":
                    value = target_value_list[i]
                    value_arr = value.split(":")
                    # print(value_arr)
                    tmp_group = df[df[target_column_list[i]].isin(value_arr)].groupby(group_list)
                    # print(tmp_group.agg({new_col: new_method_list[i]}))
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                elif target_method_list[i] == "between":
                    value = target_value_list[i]
                    value_arr = value.split(":")
                    if len(value_arr) != 2:
                        resp = HttpResponse("exception occurs.")
                        resp.headers['exception'] = "999014"
                        resp.status_code = 400
                        return resp
                    tmp_group = df[(df[target_column_list[i]] >= float(value_arr[0])) &
                                   (df[target_column_list[i]] < float(value_arr[1]))].groupby(group_list)
                    result[new_name] = tmp_group.agg({new_col: new_method_list[i]})
                else:
                    # print("method error")
                    resp = HttpResponse("exception occurs.")
                    resp.headers['exception'] = "999015"
                    resp.status_code = 400
                    return resp
        result.columns = ['_'.join(x) if isinstance(x, tuple) else x for x in result.columns.ravel()]
        # result = res_df.reset_index().to_json(force_ascii=False)
        uuid1 = set_data_to_redis(result.reset_index())
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key1 : redis 에서 가져올 데이터1의 key
# param # key2 : redis 에서 가져올 데이터2의 key
# 합칠 데이터1,2의 컬럼이 같아야 함. -> 열 방향으로 합쳐짐
# return # redis key
# ======================================================================================================================
def concat_mds(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key1 = body['key1']
            key2 = body['key2']
        else:
            key1 = request.GET.get("key1")
            key2 = request.GET.get("key2")

        df1 = get_data_from_redis(key1)
        df2 = get_data_from_redis(key2)
        if df1 is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        if df2 is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data1 = r.get(key1)
        # data2 = r.get(key2)
        # if data1 is None:
        #     resp = Response("key1 data is empty.")
        #     resp.headers['exception'] = "key1 data is empty."
        #     resp.status_code = 400
        #     return resp
        # elif data2 is None:
        #     resp = Response("key2 data is empty.")
        #     resp.headers['exception'] = "key2 data is empty."
        #     resp.status_code = 400
        #     return resp
        # else:
        #     df1 = pd.DataFrame(json.loads(data1))
        #     df2 = pd.DataFrame(json.loads(data2))
        result_df = pd.concat([df1, df2])
        uuid1 = set_data_to_redis(result_df)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key1 : redis 에서 가져올 데이터1의 key
# param # key2 : redis 에서 가져올 데이터2의 key
# data1, data2를 겹침
# return # redis key
# ======================================================================================================================
def join_mds(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key1 = body['key1']
            key2 = body['key2']
        else:
            key1 = request.GET.get("key1")
            key2 = request.GET.get("key2")

        df1 = get_data_from_redis(key1)
        if df1 is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        df2 = get_data_from_redis(key2)
        if df2 is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data1 = r.get(key1)
        # data2 = r.get(key2)
        # if data1 is None:
        #     resp = Response("key1 data is empty.")
        #     resp.headers['exception'] = "key1 data is empty."
        #     resp.status_code = 400
        #     return resp
        # elif data2 is None:
        #     resp = Response("key2 data is empty.")
        #     resp.headers['exception'] = "key2 data is empty."
        #     resp.status_code = 400
        #     return resp
        # else:
        #     df1 = pd.DataFrame(json.loads(data1))
        #     df2 = pd.DataFrame(json.loads(data2))
        result_df = df1.join(df2)
        uuid1 = set_data_to_redis(result_df)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key1 : redis 에서 가져올 데이터1의 key
# param # key2 : redis 에서 가져올 데이터2의 key
# param # column1 : 조인할 데이터1의 기준 컬럼
# param # column2 : 조인할 데이터2의 기준 컬럼
# param # method : 조인 방법 ['inner', 'outer', 'left', 'right']
# return # redis key
# ======================================================================================================================
def merge_mds(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key1 = body['key1']
            key2 = body['key2']
            column1 = body['column1']
            column2 = body['column2']
            method = body['method']
        else:
            key1 = request.GET.get("key1")
            key2 = request.GET.get("key2")
            column1 = request.GET.get("column1")
            column2 = request.GET.get("column2")
            method = request.GET.get('method')

        print(column1)
        print(column2)
        if column1 is None or column1 == "" or column1 == "null":
            resp = HttpResponse("column1 need one or more parameter")
            resp.headers['exception'] = "999016"
            resp.status_code = 400
            return resp
        else:
            column1_mod = column1.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
            column1_list = column1_mod.split(",")
        if column2 is None or column2 == "" or column2 == "null":
            resp = HttpResponse("column2 need one or more parameter")
            resp.headers['exception'] = "999017"
            resp.status_code = 400
            return resp
        else:
            column2_mod = column2.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
            column2_list = column2_mod.split(",")

        method_array = ['inner', 'outer', 'left', 'right']
        if method in method_array:
            df1 = get_data_from_redis(key1)
            if df1 is None:
                resp = HttpResponse("exception occurs.")
                resp.headers['Content-Type'] = "text/plain; charset=utf-8"
                resp.headers['exception'] = "999003"
                resp.status_code = 400
                return resp
            df2 = get_data_from_redis(key2)
            if df2 is None:
                resp = HttpResponse("exception occurs.")
                resp.headers['Content-Type'] = "text/plain; charset=utf-8"
                resp.headers['exception'] = "999003"
                resp.status_code = 400
                return resp

            merged = pd.merge(df1, df2, how=method, left_on=column1_list, right_on=column2_list)
            uuid1 = set_data_to_redis(merged)
            #return str(uuid1), 200
            return HttpResponse(str(uuid1), 200)
        else:
            resp = HttpResponse("method out of range")
            resp.headers['exception'] = "999018"
            resp.status_code = 400
            return resp
    except Exception as e:
        print(e)
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # columns : pivot 의 기준이 될 컬럼 (list 가능)
# param # method : pivot  데이터가 그룹화 될경우 나타낼 표현식
# return # redis key
# ======================================================================================================================
def pivot(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            method = body['method']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            method = request.GET.get("method")

        #method = request.args.get("method", default="first")

        method_list = ['size', 'count', 'mean', 'median', 'min', 'max', 'sum', 'prod',
                       'std', 'quantile', 'first', 'end']

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if method is None or method == "" or method == "null":
            resp = HttpResponse("method need one or more parameter")
            resp.headers['exception'] = "999011"
            resp.status_code = 400
            return resp
        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        columns_list_final = []
        col_list = set(df.columns)
        intersection_list = col_list.intersection(columns_list)
        for col in columns_list:
            if col in intersection_list:
                if col not in columns_list_final:
                    columns_list_final.append(col)
        if len(columns_list_final) == 0:
            resp = HttpResponse("Not correct columns")
            resp.headers['exception'] = "999007"
            resp.status_code = 400
            return resp
        if method in method_list:
            df2 = pd.pivot_table(df, index=columns_list_final, aggfunc=method)
            uuid1 = set_data_to_redis(df2)

            #return str(uuid1), 200
            return HttpResponse(str(uuid1), 200)
        else:
            resp = HttpResponse("method out of range")
            resp.headers['exception'] = "999019"
            resp.status_code = 400
            return resp
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# return # redis key
# 행과 열 데이터를 바꾼다.
# ======================================================================================================================
def transpose(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
        else:
            key = request.GET.get("key")

        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        df_t = df.transpose()
        uuid1 = set_data_to_redis(df_t)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
def to_int(request):
    print("to_int(정수형)=========================== ")
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        columns_list_final = []
        col_list = set(df.columns)
        intersection_list = col_list.intersection(columns_list)
        for col in columns_list:
            if col in intersection_list:
                if col not in columns_list_final:
                    columns_list_final.append(col)
        if len(columns_list_final) == 0:
            resp = HttpResponse("Not correct columns")
            resp.headers['exception'] = "999007"
            resp.status_code = 400
            return resp
        for column in columns_list_final:
            # replace ' ' to nan
            df[column] = df[column].astype(str)
            df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
            # fill nan, null, none with 0
            df[column] = df[column].fillna("0")
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                df[column] = df[column].fillna(0)
                df[column] = df[column].astype(int)
            except ValueError:
                resp = HttpResponse("Column value must be numeric value.(convert data to numeric type is failed)")
                resp.headers['exception'] = "999045"
                resp.status_code = 400
                return resp
            # df = df.astype({column: int})
        uuid1 = set_data_to_redis(df)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
def to_float(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        columns_list_final = []
        col_list = set(df.columns)
        intersection_list = col_list.intersection(columns_list)
        for col in columns_list:
            if col in intersection_list:
                if col not in columns_list_final:
                    columns_list_final.append(col)
        if len(columns_list_final) == 0:
            resp = HttpResponse("Not correct columns")
            resp.headers['exception'] = "999007"
            resp.status_code = 400
            return resp
        for column in columns_list_final:
            # replace ' ' to nan
            df[column] = df[column].astype(str)
            df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
            # fill nan, null, none with 0.0
            df[column] = df[column].fillna("0.0")
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                df[column] = df[column].fillna(0)
                df[column] = df[column].astype(float)
            except ValueError:
                resp = HttpResponse("Column value must be numeric value.(convert data to numeric type is failed)")
                resp.headers['exception'] = "999045"
                resp.status_code = 400
                return resp
            # df = df.astype({column: float})
        uuid1 = set_data_to_redis(df)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # columns : 결측값 채워 넣을 컬럼 이름
# param # fillvalue : 결측값으로 채워 넣을 값
# return # redis key
# ======================================================================================================================
def fillna(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            fill_value = body['fillvalue']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            fill_value = request.GET.get("fillvalue")

        if columns is None or columns == "" or columns == "null":
            resp = HttpResponse("columns need one or more parameter")
            resp.headers['exception'] = "999006"
            resp.status_code = 400
            return resp
        if fill_value is None or fill_value == "" or fill_value == "null":
            resp = HttpResponse("fill_value need one or more parameter")
            resp.headers['exception'] = "999020"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        columns_mod = columns.replace("[", "")
        columns_mod = columns_mod.replace("]", "")
        columns_mod = columns_mod.replace("'", "")
        columns_mod = columns_mod.replace("\"", "")
        # columns_mod = columns_mod.replace(" ", "")
        columns_list = columns_mod.split(",")
        fill_value_mod = fill_value.replace("[", "")
        fill_value_mod = fill_value_mod.replace("]", "")
        fill_value_mod = fill_value_mod.replace("'", "")
        fill_value_mod = fill_value_mod.replace("\"", "")
        # fill_value_mod = fill_value_mod.replace(" ", "")
        fill_value_list = fill_value_mod.split(",")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        columns_list_final = []
        col_list = set(df.columns)
        intersection_list = col_list.intersection(columns_list)
        for col in columns_list:
            if col in intersection_list:
                if col not in columns_list_final:
                    columns_list_final.append(col)
        if len(columns_list_final) == 0:
            resp = HttpResponse("Not correct columns")
            resp.headers['exception'] = "999007"
            resp.status_code = 400
            return resp
        if len(columns_list_final) != len(fill_value_list):
            resp = HttpResponse("Not correct columns or not same length with columns_list and fill_value list")
            resp.headers['exception'] = "999021"
            resp.status_code = 400
            return resp
        for i, column in enumerate(columns_list_final):
            # print(df[column].to_string())
            df[column] = df[column].astype(str)
            # df[column] = df[column].replace('None', np.nan, inplace=True)
            df[column] = df[column].replace('None', np.nan)
            df[column] = df[column].replace('nan', np.nan)
            # print(df[column].to_string())
            # print(fill_value_list[i])
            df[column] = df[column].fillna(value=fill_value_list[i])
            # print(df[column].to_string())
            df[column] = df[column].astype('category')
        uuid1 = set_data_to_redis(df)

        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # formula : 사칙연산 수식
# return # redis key
# ======================================================================================================================
def calculator(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            formula = body['formula']
            col_name = body['column']
        else:
            key = request.GET.get("key")
            formula = request.GET.get("formula")
            col_name = request.GET.get("column")

        #formula = request.args.get("formula", default=None)
        #col_name = request.args.get("column", default=None)
        #logger.debug(formula)

        formula_array = []
        variable = ''
        formula_operator = []
        postfix = []
        postfix_array = []
        temp_number = 0
        num = {}

        if formula is None or formula == "" or formula == "null":
            resp = HttpResponse("Formula is not correct")
            resp.headers['exception'] = "999047"
            resp.status_code = 400
            return resp
        else:
            if formula.find("+") or formula.find(" "):
                formula = formula.replace(" ", "%2b")
            if formula.find("/"):
                formula = formula.replace("/", "%2f")
        formula = parse.unquote(formula)  # decoding

        # 연산식 분리 (피연산자, 연산자) -> 연산식 배열화
        for i in formula:
            if i in '()+-*/':
                if i == '-' and variable == '':
                    variable += i
                else:
                    if i in '()+-*/':
                        if i == '(':
                            formula_array.append(i)
                        elif variable != '':
                            formula_array.append(variable)
                            formula_array.append(i)
                            variable = ''
                        else:
                            formula_array.append(i)
            else:
                variable += i
        if variable != '':
            formula_array.append(variable)
        df = get_data_from_redis(key)
        df2 = df.copy()
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # print(formula_array)
        prec = {'*': 0, '/': 0, '+': 1, '-': 1, '(': 3, ')': 3}

        # 중위연산식 -> 후위연산식
        for i in formula_array:
            if i == '(' or i == ')' or i == '+' or i == '-' or i == '*' or i == '/':
                if i == '(':
                    formula_operator.append(i)
                elif i == ')':
                    for k in formula_operator:
                        if formula_operator[-1] == '(':
                            formula_operator.pop()
                            break
                        else:
                            postfix.append(formula_operator.pop())
                else:
                    if not formula_operator:
                        formula_operator.append(i)
                    else:
                        if prec[i] < prec[formula_operator[-1]]:
                            formula_operator.append(i)
                        else:
                            postfix.append(formula_operator.pop())
                            formula_operator.append(i)
            else:
                postfix.append(i)
        for j in reversed(formula_operator):
            if j == '(' or j == ')':
                formula_operator.pop()
            else:
                postfix.append(formula_operator.pop())
        try:
            # 후위연산식 계산
            for i in postfix:
                if i == '+' or i == '-' or i == '*' or i == '/':
                    temp_number += 1
                    for z in range(2, 0, -1):
                        try:
                            if postfix_array[-1].isdigit():
                                num[z] = int(postfix_array.pop())
                            elif float(postfix_array[-1]):
                                num[z] = float(postfix_array.pop())
                        except ValueError:
                            num[z] = pd.Series(df[postfix_array.pop()])
                        except IndexError:
                            resp = HttpResponse("formula is wrong")
                            resp.headers['exception'] = "999047"
                            resp.status_code = 400
                            return resp
                    if i == '+':
                        df["temp" + str(temp_number)] = num[1] + num[2]
                    elif i == '-':
                        df["temp" + str(temp_number)] = num[1] - num[2]
                    elif i == '*':
                        df["temp" + str(temp_number)] = num[1] * num[2]
                    elif i == '/':
                        try:
                            df["temp" + str(temp_number)] = num[1] / num[2]
                        except ZeroDivisionError:
                            resp = HttpResponse("zero division")
                            resp.headers['exception'] = "999048"
                            resp.status_code = 400
                            return resp
                    else:
                        pass
                    postfix_array.append("temp" + str(temp_number))
                    # print(df["temp"+str(temp_number)])
                    # print(postfix_array)
                else:
                    postfix_array.append(i)
                    if set([i]).issubset(df.columns) is True:
                        if is_integer_dtype(pd.Series(df[i])) is True or is_float_dtype(pd.Series(df[i])) is True:
                            pass
                        elif is_categorical_dtype(pd.Series(df[i])) is True or is_object_dtype(
                                pd.Series(df[i])) is True:
                            try:
                                df[i] = df[i].astype('float')
                            except ValueError:
                                resp = HttpResponse("column type is category or object (can't be type changed)")
                                resp.headers['exception'] = "column type is category or object (can't be type changed)"
                                resp.status_code = 400
                                return resp
                        elif is_datetime64_dtype(pd.Series(df[i])) is True:
                            resp = HttpResponse("column is datetime")
                            resp.headers['exception'] = "column is datetime"
                            resp.status_code = 400
                            return resp
                    else:
                        try:
                            if postfix_array[-1].isdigit():
                                pass
                            elif float(postfix_array[-1]):
                                pass
                        except ValueError:
                            if set([i]).issubset(df.columns) is False:
                                resp = HttpResponse("column doesn't exist")
                                resp.headers['exception'] = "column doesn't exist"
                                resp.status_code = 400
                                return resp
            if len(postfix_array) != 1:
                resp = HttpResponse("formula is wrong")
                resp.headers['exception'] = "formula is wrong"
                resp.status_code = 400
                return resp
        except Exception as e:
            resp = HttpResponse("exception occurs.")
            resp.headers['exception'] = str(e)
            resp.status_code = 400
            return resp
        # print(df.head())
        # print(postfix_array)
        # df['result'] = df["temp"+str(temp_number)]
        df.rename(columns={postfix_array.pop(): 'result'}, inplace=True)
        if col_name is not None and col_name != "" and col_name != "null":
            df2[col_name] = df['result']
        else:
            df2['result'] = df['result']
        # df2 = df['result']
        print(df2.head())
        uuid1 = set_data_to_redis(df2)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# param # columns :  데이터에서 선택할 컬럼 리스트 ([data1, data2, data3 ... ] 형식)
# param # maxword : 워드클라우드에 출력될 최대 단어 수
# return # redis key
# ======================================================================================================================
def wordcloud(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            columns = body['columns']
            maxword = body['max']
        else:
            key = request.GET.get("key")
            columns = request.GET.get("columns")
            maxword = request.GET.get("max")

        #maxword = request.args.get("max", default=2000, type=int)

        column_list = []
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        if columns is None or columns == "null" or columns == "":
            for i in df.columns.values:
                column_list.append(i)
        else:
            columns_mod = columns.replace("[", "")
            columns_mod = columns_mod.replace("]", "")
            columns_mod = columns_mod.replace("'", "")
            columns_mod = columns_mod.replace("\"", "")
            # columns_mod = columns_mod.replace(" ", "")
            column_list = columns_mod.split(",")
        if maxword is None or maxword <= 0:
            maxword = 2000
        data = {}
        temp2 = []
        while column_list:
            column = column_list.pop()
            if set([column]).issubset(df.columns) is True:
                if is_categorical_dtype(pd.Series(df[column])) is True or \
                        is_object_dtype(pd.Series(df[column])) is True:
                    df[column] = df[column].astype(str)
                    temp = df[column].value_counts().to_dict()
                    data.update(temp)
                    for i in df[column].items():
                        temp1 = i[1].split()
                        for z in temp1:
                            temp2.append(z)
                    temp3 = pd.Series(temp2).value_counts().to_dict()
                    data.update(temp3)
                elif is_integer_dtype(pd.Series(df[column])) is True or is_float_dtype(pd.Series(df[column])) is True:
                    pass
                    # resp = Response("column type is int or float")
                    # resp.headers['exception'] = "column type is datetime"
                    # resp.status_code = 400
                    # return resp
                elif is_datetime64_dtype(pd.Series(df[column])) is True:
                    pass
                    # resp = Response("column type is datetime")
                    # resp.headers['exception'] = "column type is datetime"
                    # resp.status_code = 400
                    # return resp
            else:
                resp = HttpResponse("column doesn't exist")
                resp.headers['exception'] = "column doesn't exist"
                resp.status_code = 400
                return resp

        # print("max word count time", time.time() - start)
        if data is None:
            resp = HttpResponse("column type is not category, object")
            resp.headers['exception'] = "column type is not category, object"
            resp.status_code = 400
            return resp
        maxCount = max(data.values())
        # maskArray = np.array(Image.open('image3.png'))
        # font_path='잘풀리는오늘 Medium.ttf',
        wordcloud = WordCloud(
            background_color='white',
            width=800, height=800, max_words=int(maxword),
            # mask=maskArray,
            max_font_size=35
        ).generate_from_frequencies(data)
        # plt.figure(figsize=(15, 15))
        # # image_colors = ImageColorGenerator(maskArray)
        # # plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        # plt.imshow(wordcloud, interpolation="bilinear")
        # plt.axis("off")
        # plt.savefig(uuid1+'.png')
        # plt.close()
        df2 = pd.DataFrame(list(wordcloud.words_.items()), columns=['value', 'value_count'])
        # print(df2.head().to_string())
        df2['value_count'] = df2['value_count'].mul(maxCount).astype(int)
        # print(df2.head().to_string())
        uuid1 = set_data_to_redis(df2)
        # print("process time : ", time.time() - start)
        #return str(uuid1), 200
        return HttpResponse(str(uuid1), 200)
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# PIE CHART
# param # key : redis 에서 가져올 데이터의 key
# param # group : group by 할 컬럼 (1 or N) N일때는 ','로 구분 (ex - SHREG_CD, JUNG_CD)
# param # columns : 그룹화 하여 나타낼 컬럼 (1 or N) N일때는 ','로 구분 (ex - SHREG_CD, JUNG_CD)
# param # method : 그룹화 하여 보여줄 값의 함수 (sum, count, min, max, mean, std, first, last, quantile, var, median)
# param # columntoint : columns 를 int로 변환할건지 여부 ( default : false ) sum, mean 같은거 하려면 변환해줘야 함.
# return # uuid
# ======================================================================================================================
def get_chart_pie(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
            group = body['group']
            column = body['column']
            method = body['method']
        else:
            key = request.GET.get("key")
            group = request.GET.get("group")
            column = request.GET.get("column")
            method = request.GET.get("method")

        if group is None or group == "" or group == "null":
            resp = HttpResponse("group need one or more parameter")
            resp.headers['exception'] = "999022"
            resp.status_code = 400
            return resp
        if column is None or column == "" or column == "null":
            resp = HttpResponse("999005")
            resp.headers['exception'] = "999005"
            resp.status_code = 400
            return resp
        if method is None or method == "" or method == "null":
            resp = HttpResponse("method need one or more parameter")
            resp.headers['exception'] = "999011"
            resp.status_code = 400
            return resp
        # column_to_int = request.args.get("columntoint")
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        # column 이름과 method 담을 딕셔너리 f
        f = dict()
        f[column] = [method]
        # object 나 str 타입의 컬럽은 int형 타입으로 형변환 해줌. 안하면 산술 계산시 예상과 다른 값 나옴.
        # df[column] = df[column].fillna("0")
        df[column] = df[column].astype(str)
        df[column] = df[column].fillna(value="0")
        # df[column] = df[column].astype('category')
        df = df.astype({column: int})
        # 결과 컬럼 이름
        result_column = column + "_" + method
        result_df = df.groupby(group)
        res_df = result_df.agg(f)
        res_df.columns = ['_'.join(x) if isinstance(x, tuple) else x for x in res_df.columns.ravel()]
        result = res_df.reset_index().to_json(orient='records', force_ascii=False)
        df2 = pd.DataFrame(json.loads(result))
        total = df2[result_column].sum()
        df2['rate'] = df2[result_column].apply(lambda x: round((x / total) * 100, 2))
        result2 = df2.reset_index().to_json(orient='records', force_ascii=False)
        # uuid1 = get_uuid()
        # r.set(uuid1, result)
        #return str(result2), 200
        return HttpResponse(str(result2), 200)
    except Exception as e:
        print(e)
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# return # CSV file stream
# file download
# ======================================================================================================================
def download(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
        else:
            key = request.GET.get("key")

        output_stream = StringIO()
        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        df.to_csv(output_stream)
        response = HttpResponse(output_stream.getvalue(),
                            mimetype='text/csv',
                            content_type='application/octet-stream')
        response.headers["Content-Disposition"] = "attachment; filename=post_export.csv"
        return response
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp


# ======================================================================================================================
# param # key : redis 에서 가져올 데이터의 key
# return # CSV file stream
# file download
# ======================================================================================================================
def download_excel(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            key = body['key']
        else:
            key = request.GET.get("key")

        df = get_data_from_redis(key)
        if df is None:
            resp = HttpResponse("exception occurs.")
            resp.headers['Content-Type'] = "text/plain; charset=utf-8"
            resp.headers['exception'] = "999003"
            resp.status_code = 400
            return resp
        print(df)
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        # data = r.get(key)
        # if data is not None:
        #     df = pd.DataFrame(json.loads(data))
        df.to_excel(writer, sheet_name="sheet1")
        workbook = writer.book
        worksheet = writer.sheets["Sheet_1"]

        # the writer has done its job
        writer.close()

        # go back to the beginning of the stream
        output.seek(0)
        # response = Response(output_stream.getvalue(),
        #                     content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        # response.headers["Content-Disposition"] = "attachment; filename=post_export.xlsx"
        #return send_file(output, attachment_filename="testing.xlsx", as_attachment=True)
        response = HttpResponse(open(output), content_type="application/json")
        response['Content-Disposition'] = 'attachment; filename="testing.xlsx"'
        return response
    except Exception as e:
        resp = HttpResponse("exception occurs.")
        resp.headers['exception'] = str(e)
        resp.status_code = 400
        return resp



