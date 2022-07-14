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
from pip._internal.utils.parallel import TIMEOUT
from pymongo import MongoClient


mongo_host = "localhost"
mongo_port = 27017

client = MongoClient(mongo_host, int(mongo_port))

def tenSecond():
    for i in range(10):
        sys.stdout.write('\r{} sec'.format(i + 1))
        time.sleep(1)
    print('\nDone!!')


def do(a):
    tenSecond()

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

        xgb_ml = pickle.load(open(script_dir + '\\stdShregPredict', 'rb'))
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


