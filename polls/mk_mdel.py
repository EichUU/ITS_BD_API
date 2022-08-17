from pathlib import Path

import cx_Oracle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import shap
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import base64
import json
import os
import sys
import time
import urllib
from multiprocessing import Process

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from django.http import HttpResponse
# from pip._internal.utils.parallel import TIMEOUT
from pymongo import MongoClient

from dbcon.connDB import get_conn

mongo_host = "192.168.0.59"
mongo_port = 27017

client = MongoClient(mongo_host, int(mongo_port))

def mkMdel(request):
    print("mkMdel 호출")
    body_unicode = request.body.decode('utf-8')

    try:
        if len(body_unicode) != 0:
            body = json.loads(body_unicode)
            # 모델코드
            mCd = body['mCd']
            # 예측방법
            mThd = body['mThd']
            # 모델이름
            mName = body['mName']
            # 최적화유무
            mOpt = body['mOpt']
            # 예측결과물분류
            mClas = body['mClas']
            # DB종류
            tUser = body['tUser']
            # DB종류별 유저
            tUserDetail = ['tUserDetail']
            # 테이블 이름
            tName = body['tName']

        else:
            # 모델코드
            mCd = request.GET.get("mCd")
            # 예측방법
            mThd = request.GET.get("mThd")
            # 모델이름
            mName = request.GET.get("mName")
            # 최적화유무
            mOpt = request.GET.get("mOpt")
            # 예측결과물분류
            mClas = request.GET.get("mClas")
            # DB종류
            tUser = request.GET.get("tUser")
            # DB종류별 유저
            tUserDetail = request.GET.get("tUserDetail")
            # 테이블 이름
            tName = request.GET.get("tName")

            print(1)
        if tUser == 'NoSql':
            db = client[tUserDetail]
            collection = db[tName]
            # 중복데이터가 많은 관계로 마지막 학기의 마지막 주차만 들고오는 중
            # 모든 데이터를 넣으면 머신의 속도도 굉장히 느리다.
            res = collection.find({'YY': '2022', 'SHTM': '1', 'WEEK': '16'}, {'_id': 0})
            totalData = pd.DataFrame(list(res))

            print(2)
            # 사용되는 컬럼과 종속변수 컬럼을 들고온다.
            userCol = "SELECT PRD_COLUMN_NM FROM TPBPA030_D WHERE PRD_MDEL_CD = '" + mCd + "' AND PRD_MTHD ='" + mThd + "' AND USE_YN = 'Y' AND PRD_COLUMN_TARGET = 'N'"

            targetCol = "SELECT PRD_COLUMN_NM FROM TPBPA030_D WHERE PRD_MDEL_CD = '" + mCd + "' AND PRD_MTHD ='" + mThd + "' AND PRD_COLUMN_TARGET = 'Y'"

            print(3)
            cur = get_conn("oracle", "XE", "BDAS", "BDAS12#$", "mapsco.kr:1531/XE", "")

            print(4)

            resultUse = cur.execute(userCol)
            # 독립변수 컬럼
            useDf = pd.DataFrame(resultUse)
            useList = list(useDf.iloc[:, 0])

            resultTarget = cur.execute(targetCol)
            # 종속변수 컬럼
            targetDf = pd.DataFrame(resultTarget)
            targetList = list(targetDf.iloc[:, 0])

            cur.close()

            totalData['SHREG_GB'] = ((totalData['SHREG_GB'] != "B30005") | (totalData['RESN'] == "B19005") | (
                    totalData['RESN'] == "B19006") | (
                                             totalData['RESN2'] == "B23006")).apply(lambda x: '1' if x is True else "0")

            print(5)

            # 인코딩 작업
            # 정수가 아닌 오브젝트일 때, 카테고리로 지정하며 코드를 부여하고 있다
            for i in totalData.columns:
                if totalData[i].dtypes == "object":
                    totalData[i] = totalData[i].astype('category').cat.codes

            x_data = totalData[useList]
            y_data = totalData[targetList]

            print(6)
            print(x_data)
            print(y_data)

            resultList = []

            if mThd == "BDK001":
                if mOpt == 'Y':
                    resultList = mkOptXgb(x_data, y_data, mName)
                else:
                    resultList = mkXgboost(x_data, y_data, mName)

            elif mThd == "BDK002":
                if mOpt == 'Y':
                    resultList = mkOptRf(x_data, y_data, mName)
                else:
                    resultList = mkRandomForest(x_data, y_data, mName)

            elif mThd == "BDK003":
                if mOpt == 'Y':
                    resultList = mkKnn(x_data, y_data, mName)
                else:
                    resultList = mkOptKnn(x_data, y_data, mName)

            elif mThd == "BDK004":
                if mOpt == 'Y':
                    resultList = mkSvm(x_data, y_data, mName)
                else:
                    resultList = mkOptSvm(x_data, y_data, mName)

            dbuser = "BDAS"
            dbpwd = "BDAS12#$"
            dbhost = "mapsco.kr:1531/XE"

            prdConn = cx_Oracle.connect(user=dbuser, password=dbpwd, dsn=dbhost)
            cur = prdConn.cursor()

            updateSql = """UPDATE TPBPA060_M SET
                               PRD_MDEL_STATE   = '1'
                             , PRD_ACC          = '""" + str(resultList['acc']) + """'
                             , PRD_PRE          = '""" + str(resultList['pre']) + """'
                             , PRD_REC          = '""" + str(resultList['rec']) + """'
                             , PRD_F1           = '""" + str(resultList['f1']) + """'
                             , PRD_RSLT_SHAP    = '""" + str(resultList['shap']) + """'
                             , PRD_RSLT_MATRIX  = '""" + str(resultList['mat']) + """'
                             , PRD_RSLT_ROC     = '""" + str(resultList['roc']) + """'
                            WHERE PRD_MDEL_NM = '""" + mName + """'"""
            cur.execute(updateSql)
            prdConn.commit()

        # 다른 DB를 사용하게 되면 추가할 것
        elif tUser == 'etc':
            resp = HttpResponse(status=400)
            return resp

    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp

    return HttpResponse(status=200)


def mkXgboost(xData, yData, mName):
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=77)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=77)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        min_child_weight=1,
        learning_rate=0.1,
        colsample_bytree=0.1,
        objective='binary:logistic'
    )

    evals = [(x_tr, y_tr), (x_val, y_val)]

    model.fit(x_tr, y_tr,
              early_stopping_rounds=50,
              eval_metric='error',
              eval_set=evals,
              verbose=True
              )

    prds = model.predict(x_test)

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    explainer = shap.Explainer(model)
    shap_value = explainer(x_tr)
    # shap.plots.beeswarm(shap_value, show=False)
    shap.summary_plot(shap_value, x_tr, show=False)
    # 저장할 위치
    # 바탕화면/plot/cm
    # home_path = os.path.expanduser('~')

    script_dir = str(Path(os.path.dirname(__file__)).parent)
    window_path = os.path.join(script_dir, 'data/img/shap/')

    shapFName = mName + '_shap.png'
    if not os.path.isdir(window_path):
        os.makedirs(window_path)
    plt.savefig(window_path + shapFName)
    plt.close()

    # confusion matrix 그래프
    cmFName = mkConfusion(y_test, prds, model, mName)

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    rocFName = plot_roc_curve(fper, tper, 0.0, mName)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': shapFName, 'mat': cmFName, 'roc': rocFName}


def mkRandomForest(xData, yData, mName):
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=77)

    model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_features='auto',
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=2,
        verbose=2
    )

    model.fit(x_train, y_train)

    prds = model.predict(x_test)

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    # explainer = shap.TreeExplainer(model)
    # shap_value = explainer.shap_values(x_train)
    # shap_obj = explainer(x_train)
    # shap.summary_plot(
    #     shap_values=shap_value,
    #     features=x_train,
    #     plot_size=(6, 6)
    # )

    # confusion matrix 그래프
    cmFName = mkConfusion(y_test, prds, model, mName)

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    rocFName = plot_roc_curve(fper, tper, 0.0, mName)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': cmFName, 'roc': rocFName}


def mkKnn(xData, yData, mName):
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=77)

    model = KNeighborsClassifier(
        n_neighbors=1,
        weights='distance',
        metric='manhattan'
    )

    model.fit(x_train, y_train)

    prds = model.predict(x_test)

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    # explainer = shap.TreeExplainer(model)
    # shap_value = explainer.shap_values(x_train)
    # shap_obj = explainer(x_train)
    # shap.summary_plot(
    #     shap_values=shap_value,
    #     features=x_train,
    #     plot_size=(6, 6)
    # )

    # confusion matrix 그래프
    cmFName = mkConfusion(y_test, prds, model, mName)

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    rocFName = plot_roc_curve(fper, tper, 0.0, mName)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': cmFName, 'roc': rocFName}


def mkSvm(xData, yData, mName):
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=77)

    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma=0.1,
        verbose=2
    )

    model.fit(x_train, y_train)

    prds = model.predict(x_test)

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    # explainer = shap.TreeExplainer(model)
    # shap_value = explainer.shap_values(x_train)
    # shap_obj = explainer(x_train)
    # shap.summary_plot(
    #     shap_values=shap_value,
    #     features=x_train,
    #     plot_size=(6, 6)
    # )

    # confusion matrix 그래프
    cmFName = mkConfusion(y_test, prds, model, mName)

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    rocFName = plot_roc_curve(fper, tper, 0.0, mName)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': cmFName, 'roc': rocFName}



def mkOptXgb(xData, yData, mName):
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: xgbOpt(trial, xData, yData), n_trials=3)

    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=156)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=156)

    model = XGBClassifier(
        n_estimators=study.best_params.get('n_estimators'),
        max_depth=study.best_params.get('max_depth'),
        min_child_weight=study.best_params.get('min_child_weight'),
        learning_rate=study.best_params.get('learning_rate'),
        colsample_bytree=study.best_params.get('colsample_bytree'),
        objective='binary:logistic'
    )

    evals = [(x_tr, y_tr), (x_val, y_val)]

    model.fit(
        x_tr, y_tr,
        early_stopping_rounds=50,
        eval_metric='rmse',
        eval_set=evals,
        verbose=True
    )

    prds = model.predict(x_test)

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    # explainer = shap.Explainer(model)
    # shap_value = explainer(x_tr)
    # shap.plots.beeswarm(shap_value)
    # SHAP value 그래프
    explainer = shap.Explainer(model)
    shap_value = explainer(x_tr)
    # shap.plots.beeswarm(shap_value, show=False)
    shap.summary_plot(shap_value, x_tr, show=False)
    # 저장할 위치
    # 바탕화면/plot/cm

    script_dir = str(Path(os.path.dirname(__file__)).parent)
    window_path = os.path.join(script_dir, 'data/img/shap/')

    # window_path = '/img/shap/'
    shapFName = mName + '_shap.png'
    if not os.path.isdir(window_path):
        os.makedirs(window_path)
    plt.savefig(window_path + shapFName)
    plt.close()

    # confusion matrix 그래프
    cmFName = mkConfusion(y_test, prds, model, mName)

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    rocFName = plot_roc_curve(fper, tper, 0.0, mName)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': shapFName, 'mat': cmFName, 'roc': rocFName}


def xgbOpt(trial: Trial, xData, yData):
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=156)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=156)

    search_space = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 2),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.0000001, 0.2),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.0000001, 1)
    }

    evals = [(x_tr, y_tr), (x_val, y_val)]

    model = XGBClassifier(**search_space)
    model.fit(
        x_tr, y_tr,
        early_stopping_rounds=50,
        eval_metric='rmse',
        eval_set=evals,
        verbose=True
    )

    preds = model.predict(x_test)

    f1 = f1_score(y_test, preds)

    return f1


def mkOptRf(xData, yData, mName):
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: rfOpt(trial, xData, yData), n_trials=3)

    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=156)

    model = RandomForestClassifier(
        n_estimators=study.best_params.get('n_estimators'),
        criterion='gini',
        max_features='auto',
        max_depth=study.best_params.get('max_depth'),
        min_samples_leaf=study.best_params.get('min_samples_leaf'),
        min_samples_split=study.best_params.get('min_samples_split'),
        verbose=2
    )

    model.fit(x_train, y_train)

    prds = model.predict(x_test)

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    # explainer = shap.TreeExplainer(model)
    # shap_value = explainer.shap_values(x_train)
    # shap_obj = explainer(x_train)
    # shap.summary_plot(
    #     shap_values=shap_value,
    #     features=x_train,
    #     plot_size=(6, 6)
    # )

    # confusion matrix 그래프
    cmFName = mkConfusion(y_test, prds, model, mName)

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    rocFName = plot_roc_curve(fper, tper, 0.0, mName)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': cmFName, 'roc': rocFName}


def rfOpt(trial: Trial, xData, yData):
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=156)
    search_space = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2),
        'min_samples_split': trial.suggest_loguniform('min_samples_split', 0.0000001, 1.0)
    }

    model = RandomForestClassifier(
        **search_space,
        criterion='gini',
        max_features='auto',
        verbose=2
    )

    model.fit(x_train, y_train)

    preds = model.predict(x_test)

    f1 = f1_score(y_test, preds)

    return f1


def mkOptKnn(xData, yData, mName):
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: knnOpt(trial, xData, yData), n_trials=3)

    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=156)

    model = KNeighborsClassifier(
        n_neighbors=study.best_params.get('n_neighbors'),
        weights=study.best_params.get('weights'),
        metric=study.best_params.get('metric')
    )

    model.fit(x_train, y_train)

    prds = model.predict(x_test)

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    # explainer = shap.TreeExplainer(model)
    # shap_value = explainer.shap_values(x_train)
    # shap_obj = explainer(x_train)
    # shap.summary_plot(
    #     shap_values=shap_value,
    #     features=x_train,
    #     plot_size=(6, 6)
    # )

    # confusion matrix 그래프
    cmFName = mkConfusion(y_test, prds, model, mName)

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    rocFName = plot_roc_curve(fper, tper, 0.0, mName)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': cmFName, 'roc': rocFName}


def knnOpt(trial: Trial, xData, yData):
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=156)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=156)

    search_space = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 1000),
        'weights': trial.suggest_categorical('weights', ["uniform", "distance"]),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    }

    model = KNeighborsClassifier(**search_space)
    model.fit(x_tr, y_tr)

    preds = model.predict(x_test)

    f1 = f1_score(y_test, preds)

    return f1


def mkOptSvm(xData, yData, mName):
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: svmOpt(trial, xData, yData), n_trials=3)

    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=156)

    model = SVC(
        C=study.best_params.get('C'),
        kernel='rbf',
        gamma=study.best_params.get('gamma'),
        verbose=2
    )

    model.fit(x_train, y_train)

    prds = model.predict(x_test)

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    # explainer = shap.TreeExplainer(model)
    # shap_value = explainer.shap_values(x_train)
    # shap_obj = explainer(x_train)
    # shap.summary_plot(
    #     shap_values=shap_value,
    #     features=x_train,
    #     plot_size=(6, 6)
    # )

    # confusion matrix 그래프
    cmFName = mkConfusion(y_test, prds, model, mName)

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    rocFName = plot_roc_curve(fper, tper, 0.0, mName)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': cmFName, 'roc': rocFName}


def svmOpt(trial: Trial, xData, yData):
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3, random_state=156)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=156)

    search_space = {
        'C': trial.suggest_loguniform('C', 0.1, 1000),
        'gamma': trial.suggest_loguniform('gamma', 0.1, 100),
        'kernel': trial.suggest_categorical('kernel', ['sigmoid', 'rbf'])
    }

    model = SVC(
        **search_space,
        verbose=2
    )
    model.fit(x_tr, y_tr)

    preds = model.predict(x_test)

    f1 = f1_score(y_test, preds)

    return f1


# ROC((Receiver Operating Characteristic) curve 그리는 함수
def plot_roc_curve(fper, tper, title, mName):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve('+ str(title) + ')')
    plt.legend()
    # 저장할 위치
    # 바탕화면/plot/cm

    script_dir = str(Path(os.path.dirname(__file__)).parent)
    window_path = os.path.join(script_dir, 'data/img/roc/')

    rocFName = mName + '_roc.png'

    if not os.path.isdir(window_path):
        os.makedirs(window_path)

    plt.savefig(window_path + rocFName)



    plt.close()
    return rocFName


def mkConfusion(y_test, prds, model, mName):
    # confusion matrix 그래프
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    # 저장할 위치
    # 바탕화면/plot/cm

    script_dir = str(Path(os.path.dirname(__file__)).parent)
    window_path = os.path.join(script_dir, 'data/img/cm/')

    # window_path = '/img/cm/'
    cmFName = mName + '_cm.png'

    print(window_path + cmFName)

    if not os.path.isdir(window_path):
        os.makedirs(window_path)

    plt.savefig(window_path + cmFName)
    plt.close()
    return cmFName
