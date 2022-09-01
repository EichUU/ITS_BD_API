from django.test import TestCase

# Create your tests here.

from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import shap
import pickle

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
from tqdm import tqdm
from multiprocessing import Process

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from django.http import HttpResponse
# from pip._internal.utils.parallel import TIMEOUT
from pymongo import MongoClient

from dbcon.connDB import get_conn
from polls.mk_mdel import mkConfusion

mongo_host = "192.168.0.59"
mongo_port = 27017

client = MongoClient(mongo_host, int(mongo_port))


def mkMdel():
    print("mkMdel 호출")

    try:
        # 모델코드
        mCd = '000001'
        # 예측방법
        mThd = 'BDK001'
        # 모델이름
        mName = '000001_BDK001_000001'
        # 최적화유무
        mOpt = 'N'
        # 예측결과물분류
        mClas = 'BDL001'
        # DB종류
        tUser = 'NoSql'
        # DB종류별 유저
        tUserDetail = 'BDAS'
        # 테이블 이름
        tName = 'MCBPA010_M'

        resultList = []

        print(1)
        if tUser == 'NoSql':
            db = client[tUserDetail]
            collection = db[tName]
            res = collection.find({'YY': '2022', 'SHTM': '2', 'WEEK': '1'}, {'_id': 0})
            totalData = pd.DataFrame(list(res))

            print(totalData.columns)
            print(totalData.head())

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
            print(useList)

            resultTarget = cur.execute(targetCol)
            # 종속변수 컬럼
            targetDf = pd.DataFrame(resultTarget)
            targetList = list(targetDf.iloc[0])
            print(targetList)

            totalData['SHREG_GB'] = ((totalData['SHREG_GB'] != "B30005") | (totalData['RESN'] == "B19005") | (
                    totalData['RESN'] == "B19006") | (
                                             totalData['RESN2'] == "B23006")).apply(lambda x: '1' if x is True else "0")

            print(5)
            print(totalData['SHREG_GB'])

            for i in totalData.columns:
                if totalData[i].dtypes == "object":
                    totalData[i] = totalData[i].astype('category').cat.codes

            # encodingData = pd.get_dummies(totalData)
            x_data = totalData[useList]
            y_data = totalData[targetList]
            # x_data = pd.get_dummies(X_data)
            # y_data = pd.get_dummies(Y_data)

            print(6)
            print(x_data)
            print(y_data)

            if mThd == "BDK001":
                if mOpt == 'Y':
                    resultList = mkOptXgb(x_data, y_data)
                else:
                    resultList = mkXgboost(x_data, y_data, mName)

            elif mThd == "BDK002":
                if mOpt == 'Y':
                    resultList = mkOptRf(x_data, y_data)
                else:
                    resultList = mkRandomForest(x_data, y_data)

            elif mThd == "BDK003":
                if mOpt == 'Y':
                    resultList = mkOptKnn(x_data, y_data)
                else:
                    resultList = mkKnn(x_data, y_data)

            elif mThd == "BDK004":
                if mOpt == 'Y':
                    resultList = mkOptSvm(x_data, y_data)
                else:
                    resultList = mkSvm(x_data, y_data)

        # 다른 DB를 사용하게 되면 추가할 것
        elif tUser == 'etc':
            return ""

        print(str(resultList['acc']))
        print(str(resultList['pre']))
        print(str(resultList['rec']))
        print(str(resultList['f1']))
        print(str(resultList['shap']))
        print(str(resultList['mat']))
        print(str(resultList['roc']))

    except Exception as e:
        print(e)
        return ""


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

    saveMdel(model, mName)

    prds = model.predict(x_test)
    prds_proba = model.predict_proba(x_test)
    prds_proba_positive = prds_proba[:, 1]

    acc = round(accuracy_score(y_test, prds) * 100, 3)
    pre = round(precision_score(y_test, prds) * 100, 3)
    rec = round(recall_score(y_test, prds) * 100, 3)
    f1 = round(f1_score(y_test, prds) * 100, 3)

    # SHAP value 그래프
    explainer = shap.Explainer(model)
    shap_value = explainer(x_tr)
    # shap.plots.beeswarm(shap_value)
    shap.summary_plot(shap_value, x_tr, show=False)

    # 저장할 위치
    # 바탕화면/plot/cm
    home_path = os.path.expanduser('~')
    window_path = os.path.join(home_path, 'Desktop/plot/cm/')
    fname = '01_test_01'
    if not os.path.isdir(window_path):
        os.makedirs(window_path)
    plt.savefig(window_path + fname + ".png")
    plt.close()

    # confusion matrix 그래프
    # confusion matrix 그래프
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    plt.show()

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds_proba_positive)
    plot_roc_curve(fper, tper, 0.0)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': 'image', 'roc': 'image'}


def mkRandomForest(xData, yData):
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
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    plt.show()

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    plot_roc_curve(fper, tper, 0.0)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': 'image', 'roc': 'image'}


def mkKnn(xData, yData):
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
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    plt.show()

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    plot_roc_curve(fper, tper, 0.0)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': 'image', 'roc': 'image'}


def mkSvm(xData, yData):
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
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    plt.show()

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    plot_roc_curve(fper, tper, 0.0)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': 'image', 'roc': 'image'}


def mkOptXgb(xData, yData):
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
    explainer = shap.Explainer(model)
    shap_value = explainer(x_tr)
    shap.plots.beeswarm(shap_value)

    # confusion matrix 그래프
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    plt.show()

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    plot_roc_curve(fper, tper, 0.0)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': 'image', 'roc': 'image'}


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


def mkOptRf(xData, yData):
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
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    plt.show()

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    plot_roc_curve(fper, tper, 0.0)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': 'image', 'roc': 'image'}


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


def mkOptKnn(xData, yData):
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
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    plt.show()

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    plot_roc_curve(fper, tper, 0.0)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': 'image', 'roc': 'image'}


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


def mkOptSvm(xData, yData):
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
    cm = confusion_matrix(y_test, prds, labels=model.classes_)
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plot.plot()
    plt.show()

    # ROC((Receiver Operating Characteristic) curve 그래프 그리기
    fper, tper, thresholds = roc_curve(y_test, prds)
    plot_roc_curve(fper, tper, 0.0)

    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'shap': 'image', 'mat': 'image', 'roc': 'image'}


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
def plot_roc_curve(fper, tper, title):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve(XGBoost :' + str(title) + ')')
    plt.legend()
    plt.show()


def saveMdel(model, mName):
    script_dir = str(Path(os.path.dirname(__file__)).parent)
    window_path = os.path.join(script_dir, 'data/mdel/')

    if not os.path.isdir(window_path):
        os.makedirs(window_path)

    with open(window_path + mName + '.pickle', 'wb') as fw:
        pickle.dump(model, fw)


mkMdel()
