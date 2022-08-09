import base64
import json
import os
import sys
import time
import urllib
from multiprocessing import Process

import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
from django.http import HttpResponse
# from pip._internal.utils.parallel import TIMEOUT
from pymongo import MongoClient

from dbcon.connDB import get_conn

mongo_host = "192.168.0.59"
mongo_port = 27017

client = MongoClient(mongo_host, int(mongo_port))


def selectPRDTable(request):
    print("selectPRDTable 호출")
    body_unicode = request.body.decode('utf-8')

    try:
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            # DB종류
            tUser = body['tUser']
            # DB종류별 유저
            tUserDetail = ['tUserDetail']

        else:
            # DB종류
            tUser = request.GET.get("tUser")
            # DB종류별 유저
            tUserDetail = request.GET.get("tUserDetail")

        if tUser == 'NoSql':
            db = client[tUserDetail]
            res = db.list_collection_names()

            # 여기서 컬렉션 이름들을 토대로 용어사전을 통해 각 컬렉션들의 설명을 불러온 후 df에 함께 배치 한다.
            # 분류명과 테이블 유저를 여기서 같이 붙여 줘야 한다.
            colDf = pd.DataFrame(list(res))
            colDf.columns = ['PRD_TABLE_NM']

            count = len(list(res))
            user = []
            userDetail = []
            desc = []

            for i in range(count):
                user.append(tUser)
                userDetail.append(tUserDetail)
                desc.append("")

            colDf['PRD_TABLE_USER'] = user
            colDf['PRD_TABLE_USER_DETAIL'] = userDetail
            colDf['PRD_TABLE_DESC'] = desc

            resp = HttpResponse(colDf.to_json(orient='records', force_ascii=False))
            resp.status_code = 200
            return resp

        if tUser == 'BDAS':
            # 다른 DB(공통코드에 등록된)를 사용할 때
            colDf = pd.DataFrame()
            resp = HttpResponse(colDf.to_json(orient='records', force_ascii=False))
            resp.status_code = 200
            return resp

        if tUser == 'BIG_DATA':
            # 다른 DB(공통코드에 등록된)를 사용할 때
            colDf = pd.DataFrame()
            resp = HttpResponse(colDf.to_json(orient='records', force_ascii=False))
            resp.status_code = 200
            return resp

        if tUser == 'UPMS':
            # 다른 DB(공통코드에 등록된)를 사용할 때
            colDf = pd.DataFrame()
            resp = HttpResponse(colDf.to_json(orient='records', force_ascii=False))
            resp.status_code = 200
            return resp

    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp


def insertPRDTable(request):
    print("insertPRDTable 호출")

    try:
        body_unicode = request.body.decode('utf-8')
        print(1)

        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            # DB종류
            tUser = body['tUser']
            # DB종류별 유저
            tUserDetail = ['tUserDetail']
            # 테이블 이름
            tName = body['tName']
            print(2)

        else:
            # DB종류
            tUser = request.GET.get("tUser")
            # DB종류별 유저
            tUserDetail = request.GET.get("tUserDetail")
            # 테이블 이름
            tName = request.GET.get("tName")
            print(3)

        if tUser == 'NoSql':
            db = client[tUserDetail]
            collection = db[tName]

            # 모든 행 컬럼이 같을 경우 limit을 사용하여 첫째행만 가져온다.
            res = collection.find({}, {'_id': 0}).limit(1)
            print(4)

            # 가져온 데이터를 데이터프레임 형태로 저장
            dfColumn = pd.DataFrame(list(res))
            # 데이터가 있을경우 실행하고, 없다면 404 에러
            if len(dfColumn.columns) > 0:

                # 데이터프레임 컬럼명
                arrColumns = dfColumn.columns

                strColumn = ""

                # 데이터프레임 컬럼명을 통해 in 절 text
                for i in arrColumns:
                    strColumn += "'" + i + "',"

                # 마지막 , 제거
                strColumn = strColumn.rstrip(",")
                print(5)

                # 나중에 따로 function으로 빼서 사용할 수 있음
                # 용어사전과 매칭하여 이름과 설명을 들고온다.
                sql = """SELECT DICT_COL_NM, DICT_DETAIL_NM FROM TPSYA060_M WHERE DICT_COL_NM IN (""" + strColumn + """)"""

                cur = get_conn("oracle", "XE", "BDAS", "BDAS12#$", "mapsco.kr:1531/XE", "")
                cur = cur.execute(sql)

                # 오라클 쿼리 결과를 데이터프레임으로 변경
                df = pd.DataFrame(cur)
                df.columns = ["DICT_COL_NM", "DICT_DETAIL_NM"]

                # 몽고db에서 불러온 컬럼명을 데이터프레임으로 변경.
                # 이 때, 오라클 데이터와 조인하기 위해 DICT_COL_NM 으로 만듬
                dfColumn = pd.DataFrame(arrColumns, columns=["DICT_COL_NM"])
                print(6)

                # 오라클에서 불러온 데이터가 없을 경우 COMMENT를 컬럼명으로 지정한다.
                if df.shape[0] == 0:
                    dfColumn['DICT_DETAIL_NM'] = dfColumn['DICT_COL_NM']
                    returnDf = dfColumn
                # 오라클에서 불러온 데이터와 조인
                else:
                    returnDf = pd.merge(dfColumn, df, on="DICT_COL_NM", how="left")
                    print(7)

                # 오라클에서 COMMENT가 없을경우 컬럼명으로 변경한다.
                returnDf["DICT_DETAIL_NM"] = returnDf["DICT_DETAIL_NM"].fillna(returnDf["DICT_COL_NM"])

                # 컬럼명으로 받은 field값을 순차적으로 넣어 field type을 반환 받는다.
                # typeList = pd.DataFrame(columns=['field', 'type'])
                typeList = []
                for data in arrColumns:
                    dataType = collection.aggregate(
                        [{'$project': {'_id': 0, data: {'$type': '$' + data}}}, {'$limit': 1}])
                    # typeDf = pd.DataFrame(list(dataType), columns=['field', 'type'])
                    # typeDf.columns = ['field', 'type']
                    # typeList.append(list(dataType))
                    type_dic = list(dataType)
                    type_list = type_dic[0].values()
                    type_str = list(type_list)[0]
                    typeList.append(type_str)

                returnDf['DICT_COL_TYPE'] = typeList
                print(returnDf.columns)
                returnDf.columns = ['PRD_COLUMN_NM', 'PRD_COLUMN_DESC', 'PRD_COLUMN_TYPE']

                resp = HttpResponse(returnDf.to_json(orient='records', force_ascii=False))
                resp.status_code = 200
                return resp

        if tUser == 'BDAS':
            returnDf = pd.DataFrame()
            returnDf.columns = ['PRD_COLUMN_NM', 'PRD_COLUMN_DESC']
            resp = HttpResponse(returnDf.to_json(orient='records', force_ascii=False))
            resp.status_code = 200
            return resp

        if tUser == 'BIG_DATA':
            returnDf = pd.DataFrame()
            returnDf.columns = ['PRD_COLUMN_NM', 'PRD_COLUMN_DESC']
            resp = HttpResponse(returnDf.to_json(orient='records', force_ascii=False))
            resp.status_code = 200
            return resp

        if tUser == 'UPMS':
            returnDf = pd.DataFrame()
            returnDf.columns = ['PRD_COLUMN_NM', 'PRD_COLUMN_DESC']
            resp = HttpResponse(returnDf.to_json(orient='records', force_ascii=False))
            resp.status_code = 200
            return resp

    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp


def dataCount(request):
    try:
        body_unicode = request.body.decode('utf-8')
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            # DB종류
            tUser = body['tUser']
            # DB종류별 유저
            tUserDetail = ['tUserDetail']
            # 테이블 이름
            tName = body['tName']
            print(2)

        else:
            # DB종류
            tUser = request.GET.get("tUser")
            # DB종류별 유저
            tUserDetail = request.GET.get("tUserDetail")
            # 테이블 이름
            tName = request.GET.get("tName")
            print(3)

        if tUser == 'NoSql':
            db = client[tUserDetail]
            collection = db[tName]

            # 조회한 데이터 수를 반환한다
            res = {"total_data_count": collection.count()}
            print(res)
            print(4)
            resp = HttpResponse(json.dumps(res))
            resp.status_code = 200
            return resp

    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp
