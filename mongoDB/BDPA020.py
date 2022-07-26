import base64
import json
import os
import re
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
from pip._internal.utils.parallel import TIMEOUT
from pymongo import MongoClient

from dbcon.connDB import get_conn

mongo_host = "192.168.0.59"
mongo_port = 27017

client = MongoClient(mongo_host, int(mongo_port))

def get_datas(_db_name, _collection_name):
    db = client[_db_name]
    collection = db[_collection_name]
    res = collection.find()
    print("got md", _db_name, list(res))
    return list(res)

def getStdInform(request):
    body_unicode = request.body.decode('utf-8')

    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        stdNo = body['STD_NO']
        frYy = body['FR_YY']
        frShtm = body['FR_SHTM']
        toYy = body['TO_YY']
        toShtm = body['TO_SHTM']

    else:
        stdNo = request.GET.get("STD_NO")
        frYy = request.GET.get("FR_YY")
        frShtm = request.GET.get("FR_SHTM")
        toYy = request.GET.get("TO_YY")
        toShtm = request.GET.get("TO_SHTM")

    try:
        # print(get_datas('BDAS', 'MCBPA010_M'))

        db = client['BDAS']
        collection = db['MCBPA010_M']
        res = collection.find({"STD_NO":stdNo}, {"_id":False, "SHAP_VALUE":False})
        # print("got md", 'BDAS', list(res))

        stdDf = pd.DataFrame(list(res))

        # print(stdDf[(stdDf['YY'] + stdDf['SHTM'] >= frYy + frShtm & stdDf['YY'] + stdDf['SHTM'] <= toYy + toShtm)])

        # stdDf = stdDf[((stdDf['YY'] + stdDf['SHTM'] >= frYy + frShtm) & (stdDf['YY'] + stdDf['SHTM'] <= toYy + toShtm))==True]

        # print(stdDf.to_dict(orient='records'))
        # resp = HttpResponse(stdDf.to_json(orient='records', ensure_ascii=False))
        resp = HttpResponse(stdDf.to_json(orient='records', force_ascii=False))
        resp.status_code = 200
        return resp
    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp


def getImage(request):
    body_unicode = request.body.decode('utf-8')

    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        stdNo = body['STD_NO']
        yy = body['YY']
        shtm = body['SHTM']
        week = body['WEEK']

    else:
        stdNo = request.GET.get("STD_NO")
        yy = request.GET.get("YY")
        shtm = request.GET.get("SHTM")
        week = request.GET.get("WEEK")



    try:

        db = client['BDAS']
        collection = db['MCBPA020_M']
        res = collection.find({"STD_NO": stdNo, "YY": yy, "SHTM": shtm, "WEEK": week}, {"_id":False, "STD_NO":False, "YY":False, "SHTM":False, "WEEK":False})

        df = pd.DataFrame(list(res))

        for i in df.columns:
            try:
                df[i] = df[i].astype(float)
            except Exception as e:
                print(e)

        script_dir = os.path.dirname(__file__)

        xgb_ml = pickle.load(open(script_dir + '\\stdShregPredictWithCnt', 'rb'))
        explainer = shap.TreeExplainer(xgb_ml)
        shap_values = explainer.shap_values(df)

        print(df)

        plt.rcParams["font.family"] = 'MaruBuri'
        shap.initjs()
        # dependence_plot
        # 총 13개 특성의 Shapley value를 절댓값 변환 후 각 특성마다 더함 -> np.argsort()는 작은 순서대로 정렬, 큰 순서대로 정렬하려면
        # 앞에 마이너스(-) 기호를 붙임
        shap.decision_plot(explainer.expected_value, shap_values[0, :], df.iloc[0, :], show=False)
        plt.savefig(script_dir + '\\' + str(stdNo) + '_1.png', bbox_inches='tight')
        plt.clf()

        shap.force_plot(explainer.expected_value, shap_values[0, :], df.iloc[0, :], matplotlib=True, show=False)
        plt.savefig(script_dir + '\\' + str(stdNo) + '_2.png', bbox_inches='tight')
        plt.clf()

        # summary_plot : 전체 특성들이 Shapley value 분포에 어떤 영향을 미치는지 시각화
        # y축은 각 특성을, x축은 Shapely value를 나타냄(색깔은 특성값을 나타내어 빨간색으로 갈수록 높은 값)
        # 그래프 상에서 특성은 예측에 미치는 영향력(=중요도)에 따라 정렬
        # shap.summary_plot(shap_values, test_X)


        class ShapInput(object):
            def __init__(self, expectation, shap_values, features, feat_names):
                self.base_values = expectation
                self.values = shap_values
                self.data = features
                self.feature_names = feat_names
                self.display_data = features

        shap_input = ShapInput(explainer.expected_value, shap_values[0],
                               df.iloc[0, :], feat_names=df.columns)

        shap.waterfall_plot(shap_input, show=False)
        plt.savefig(script_dir + '\\' + str(stdNo) + '_3.png', bbox_inches='tight')
        plt.clf()

        arrImage = []
        for i in range(1,4):
            with open(script_dir + '\\' + str(stdNo) + '_' + str(i) + '.png', 'rb') as img:
                base64_string = base64.b64encode(img.read())

            base64_string = urllib.parse.quote(base64_string)
            arrImage.append(base64_string)
            try:
                os.remove(script_dir + '\\' + str(stdNo) + '_' + str(i) + '.png')

            except Exception as e:
                print(e)

        resp = HttpResponse(json.dumps(arrImage))
        resp.status_code = 200
        return resp

    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp

def getColumnComment(request) :
    body_unicode = request.body.decode('utf-8')

    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        db = body['DB']
        collection = body['COLLECTION']

    else:
        db = request.GET.get("DB")
        collection = request.GET.get("COLLECTION")

    try:
        db = client[db]
        col = db[collection]

        # resp = getColumnCommentO(col)
        resp = getColumnCommentM(col)
        return resp

    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp

def getColumnCommentO(col):

    # COMMENT가 저장되어있는 테이블 전체 호출
    sql = """SELECT DICT_COL_NM, DICT_DETAIL_NM FROM TPSYA060_M """
    cur = get_conn("oracle", "XE", "BDAS", "BDAS12#$", "mapsco.kr:1531/XE", "")
    cur = cur.execute(sql)

    # 데이터프레임 선언
    returnDf = pd.DataFrame(columns=["DICT_COL_NM", "DICT_DETAIL_NM"])

    # 데이터 프레임 인덱스 변수
    index = 0

    # ORACLE 쿼리 결과 커서로 FOR문 동작
    # i : DICT_COL_NM
    # j : DICT_DETAIL_NM
    for i, j in cur.fetchall():
        try:
            # 컬럼이 존재하는지 MONGO에서 조회
            res = col.find({i: {"$exists": True}}, {'_id': 0, i: 1}).limit(1)
            # 컬럼이 존재한다면 해당 컬럼을 컬럼명과 함께 데이터프레임에 저장
            if res.count() > 0:
                returnDf.loc[index] = [list(res[0].keys())[0], j]
                index += 1
        except Exception as e:
            print(e)

    resp = HttpResponse(returnDf.to_json(orient='records', force_ascii=False))
    resp.status_code = 200
    return resp

# 기준 : mongodb(속도가 느림)
# 몽고db에서 전체 행을 읽어와 column명 추출
def getColumnCommentM(col):

    #전체 행 조회
    # res = col.find({}, {'_id': 0})

    #모든 행 컬럼이 같을경우 limit을 사용하여 첫째행만 가져온다.
    res = col.find({}, {'_id': 0}).limit(1)

    # 가져온 데이터를 데이터프레임형태로 저장
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

        sql = """SELECT DICT_COL_NM, DICT_DETAIL_NM FROM TPSYA060_M WHERE DICT_COL_NM IN (""" + strColumn + """)"""

        cur = get_conn("oracle", "XE", "BDAS", "BDAS12#$", "mapsco.kr:1531/XE", "")
        cur = cur.execute(sql)

        # 오라클 쿼리 결과를 데이터프레임으로 변경
        df = pd.DataFrame(cur)
        df.columns = ["DICT_COL_NM", "DICT_DETAIL_NM"]

        # 몽고db에서 불러온 컬럼명을 데이터프레임으로 변경.
        # 이 때, 오라클 데이터와 조인하기 위해 DICT_COL_NM 으로 만듬
        dfColumn = pd.DataFrame(arrColumns, columns=["DICT_COL_NM"])

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

        resp = HttpResponse(returnDf.to_json(orient='records', force_ascii=False))
        resp.status_code = 200
        return resp

    else:
        resp = HttpResponse("저장된 데이터가 존재하지 않습니다.")
        resp.status_code = 404
        return resp



def getMongoQuery(request):
    body_unicode = request.body.decode('utf-8')

    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        agg = str(body['AGG'])

    else:
        agg = str(request.GET.get("AGG"))

    db = client['BDAS']
    col = db['MCBPA010_M']


    try:
        print(agg)
        res = col.aggregate(json.loads(agg))

        columns = []
        res = list(res)

        if '_id' in res[0].keys():
            columns = list(res[0]['_id'].keys())

        columns += list(res[0].keys())
        columns.remove('_id')
        print(columns)

        df = pd.DataFrame(columns=columns)

        index = 0

        for i in res:
            temp = []
            for j in i:
                print(j)
                if j == '_id':
                    for k in i[j]:
                        temp.append(i[j][k])
                else:
                    temp.append(i[j])
            df.loc[index] = temp
            index += 1

        print(1)
        resp = HttpResponse(df.to_json(orient='records', force_ascii=False))
        resp.status_code = 200
        return resp

    except Exception as e:
        print(e)
        resp.status_code = 400
        return resp