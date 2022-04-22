import base64
import json
import os
import re
import urllib
import uuid

import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
import logging
import plotly.express as px
import plotly.graph_objects as go

from urllib import parse
from matplotlib import pyplot as plt, font_manager, rc
from scipy.special import expit
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
#from dython import nominal
from polls.datamds_master import get_data_from_redis
from django.http import HttpResponse
from django.shortcuts import render

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumSquare'


def analyze(request):
    body_unicode = request.body.decode('utf-8')
    redisKey = ""
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        redisKey = body['redisKey']
        analy = body['analyze']
        indep = body['independent']
        dep = body['dependent']
        depth = body['depth']
        estimators = body['estimators']
        value = body['value']
        data = body['data']
        target = body['target']
        cluster = body['cluster']
        kernel = body['kernel']
        statX = body['x']
        statY = body['y']
        statZ = body['z']
        statColor = body['color']
        statSize = body['size']
        statSymbol = body['symbol']

    else:
        redisKey = request.GET.get("redisKey")
        analy = request.GET.get("analyze")
        indep = request.GET.get("independent")
        dep = request.GET.get("dependent")
        depth = request.GET.get("depth")
        estimators = request.GET.get("estimators")
        value = request.GET.get("value")
        data = request.GET.get("data")
        target = request.GET.get("target")
        cluster = request.GET.get("cluster")
        kernel = request.GET.get("kernel")

        statX = request.GET.get("x")
        statY = request.GET.get("y")
        statZ = request.GET.get("z")
        statColor = request.GET.get("color")
        statSize = request.GET.get("size")
        statSymbol = request.GET.get("symbol")

    df = get_data_from_redis(redisKey)

    indep = re.sub('[\" + \] + \[]', '', indep)
    dep = re.sub('[\" + \] + \[]', '', dep)
    data = re.sub('[\" + \] + \[]', '', data)
    target = re.sub('[\" + \] + \[]', '', target)
    value = re.sub('[\" + \] + \[]', '', value)
    statX = re.sub('[\" + \] + \[]', '', statX)
    statY = re.sub('[\" + \] + \[]', '', statY)
    statZ = re.sub('[\" + \] + \[]', '', statZ)
    statColor = re.sub('[\" + \] + \[]', '', statColor)
    statSize = re.sub('[\" + \] + \[]', '', statSize)
    statSymbol = re.sub('[\" + \] + \[]', '', statSymbol)

    columns = [item.replace(' ', '') for item in df.columns]
    df.columns = columns
    print(columns)

    plt.clf()
    #   df = df.dropna()
    analyStat = ["BDH012", "BDH013", "BDH014", "BDH015"]
    analyPrd = ["BDH005", "BDH006", "BDH007", "BDH008"]
    analyCls = ["BDH001", "BDH002", "BDH003", "BDH004", "BDH009", "BDH010", "BDH011"]

    dfStat = pd.DataFrame()

    if analy in analyStat:
        arrayStat = statX.split(",") + statY.split(",") + statZ.split(",") + statColor.split(",") + statSize.split(",") + statSymbol.split(",")

        #리스트에서 곻백인 원소 제거
        arrayStat = [v for v in arrayStat if v]
        print(arrayStat)
        dfStat = df[arrayStat]
        print(dfStat)

    elif analy in analyCLs:
        try:
            df = df[indep.split(",") + dep.split(",")]
        except Exception as e:
            print(e)
            df = df[indep.split(",")]

        df = df.replace(r'', np.nan, regex=True)
        df = df.dropna()
        indep = indep.split(",")
        indep = df[indep]

        try:
            dep = dep.split(",")
            dep = df[dep]
            print("=============")
            print(dep)
        except Exception as e:
            print(e)

        try:
            for i in dep.columns:
                try:
                    if len(dep[i].iloc[0]) > 4:
                        dep[i] = pd.to_datetime(df[i], format='%m-%d')
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
        print(123)

        try:
            for i in indep.columns:
                try:
                    if len(indep[i].iloc[0]) > 4:
                        indep[i] = pd.to_datetime(indep[i], format='%m-%d')
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

        print(2)

    else:
        data = data.split(",")
        target = target.split(",")
        value = value.split(",")
        data = df[data]
        target = df[target]
        value = value[1::2]

    try:
        if analy == "BDH001":
            print(1)
            # 선형회귀
            res = linearRegression(indep, dep)

        elif analy == "BDH002":
            # 로지스틱회귀
            res = logisticRegression(indep, dep)

        elif analy == "BDH003":
            # 의사결정나무
            res = DecisionTree(indep, dep, depth)

        elif analy == "BDH004":
            # 의사결정나무
            res = randomForest(indep, dep, estimators, depth)

        elif analy == "BDH005":
            # K-최근접이웃
            res = KNN(data, target, value, cluster)

        elif analy == "BDH006":
            # K-평균
            res = kMeans(data, target)

        elif analy == "BDH007":
            # SVM
            res = svm(data, target, value, kernel)

        elif analy == "BDH008":
            # 나이브베이즈
            res = naive_bayes(indep, dep)

        elif analy == "BDH009":
            #           plt.figure(figsize=(15, 15))
            try:
                xlen = len(indep[indep.columns[0]].values.categories)
                ylen = len(indep[indep.columns[1]].values.categories)
                if xlen > ylen:
                    plt.figure(figsize=(xlen / 4, xlen / 4))
                else:
                    plt.figure(figsize=(ylen / 4, ylen / 4))
            except:
                plt.figure(figsize=(10, 10))
            plt.xticks(rotation=90)
            try:
                plt.rc('font', family="NanumSqure", size=5)
            except Exception as e:
                print(e)

            # plt.grid()

            try:
                y = pd.to_numeric(indep[indep.columns[1]])
                plt.ylim(y.min(), y.max())
            except:
                y = indep[indep.columns[1]]

            try:
                x = pd.to_numeric(indep[indep.columns[0]])
                plt.xlim(x.min(), x.max())
            except:
                x = indep[indep.columns[0]]

            z = pd.to_numeric(dep[dep.columns[0]])
            plt.grid()
            sns.scatterplot(x=x, y=y, size=z, sizes=(0, 400), legend=False)
            plt.grid()
            sns.set(font="NanumSquare", rc={"axes.unicode_minus": False, "font.size": 5})

            fname = str(uuid.uuid4()) + ".jpg"
            print(fname)
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'Results/bubble/')
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            plt.savefig(results_dir + fname, bbox_inches='tight')

            with open(results_dir + fname, 'rb') as img:

                base64_string = base64.b64encode(img.read())
            base64_string = urllib.parse.quote(base64_string)
            res = {'fname': base64_string}

        elif analy == "BDH010":
            try:
                print(123)
                le = LabelEncoder()
                print(indep)
                for i in indep.columns:
                    indep[i] = le.fit_transform(indep[i])
                corr_column_names = indep.columns
                #                nominal.associations(indep)
                print(indep)
                plt.figure(figsize=(len(indep.columns) * 4, len(indep.columns) * 4))
                heatmap = sns.heatmap(indep.apply(pd.to_numeric).corr(method='kendall'), cbar=True, annot=True,
                                      fmt='.3f', square='True', yticklabels=corr_column_names,
                                      xticklabels=corr_column_names)
            #                heatmap = sns.heatmap(pd.get_dummies(indep).corr(method='kendall'), cbar = True, annot = True, fmt ='.3f', square = 'True', yticklabels=corr_column_names, xticklabels=corr_column_names)
            except Exception as e:
                print(e)
                print(indep)

            fname = str(uuid.uuid4()) + ".jpg"

            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'Results/heatmap/')
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            plt.savefig(results_dir + fname, bbox_inches='tight')

            with open(results_dir + fname, 'rb') as img:

                base64_string = base64.b64encode(img.read())
            base64_string = urllib.parse.quote(base64_string)
            print(base64_string)
            res = {'fname': base64_string}

        elif analy == "BDH011":
            try:
                y = pd.to_numeric(dep[dep.columns[0]])
                plt.ylim(y.min(), y.max())
            except:
                y = dep[dep.columns[0]]

            try:
                x = pd.to_numeric(indep[indep.columns[0]])
                plt.xlim(x.min(), x.max())
            except:
                x = indep[indep.columns[0]]

            fig, ax = plt.subplots()
            print(1)
            try:
                sns.boxplot(ax=ax, x=x, y=y)
            except Exception as e:
                print(e)
            print(2)

            fname = str(uuid.uuid4()) + ".jpg"

            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'Results/box/')

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            plt.savefig(results_dir + fname, bbox_inches='tight')

            with open(results_dir + fname, 'rb') as img:
                base64_string = base64.b64encode(img.read())

            base64_string = urllib.parse.quote(base64_string)
            print(base64_string)
            res = {'fname': base64_string}
        elif analy == "BDH012":
            res = histogram(dfStat, statX)
        elif analy == "BDH013":
            res = bar(dfStat, statX, statY, statColor)
        elif analy == "BDH014":
            res = line(dfStat, statX, statY, statColor)
        elif analy == "BDH015":
            res = scatter(dfStat, statX, statY, statZ, statColor, statSize, statSymbol)
        #        print(res)
        resp = HttpResponse(str(res))
        resp.status_code = 200
    except Exception as e:
        print(e)
        resp = HttpResponse()
        resp.status_code = 404

    return resp


def linearRegression(train_x, train_y):
    sm.add_constant(train_x)
    train_x = train_x.apply(pd.to_numeric)
    train_y = train_y.apply(pd.to_numeric)

    mod = sm.OLS(train_x, train_y)
    res = mod.fit()
    fname = ""
    results_dir = ""
    if len(train_y.columns.values) == 1:
        fig = plt.figure(figsize=(8, 8))

        #        plt.xlim(train_x.min(), train_x.max())

        #        plt.ylim(train_y.min(), train_y.max())

        plt.scatter(train_y, train_x)
        plt.ylabel(train_x.columns.values[0])
        plt.xlabel(train_y.columns.values[0])
        plt.plot(train_y, res.fittedvalues, color='red')

        #        ax.scatter(train_y, train_x)
        #        dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
        #        ax.xaxis.set_major_formatter(dateFmt)

        fname = str(uuid.uuid4()) + ".jpg"

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Results/linearRegression/')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(results_dir + fname)
        # plt.show()

    if len(train_y.columns.values) == 2:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        # print(1)
        # print(train_x.shape)
        # print(train_y[train_y.columns.values[0]].values.reshape(-1,1).shape)
        # print(train_y[train_y.columns.values[1]].values.reshape(-1,1).shape)
        #
        # print(2)
        x_pred = np.linspace(train_y[train_y.columns.values[0]].min(), train_y[train_y.columns.values[0]].max(), 10)
        y_pred = np.linspace(train_y[train_y.columns.values[1]].min(), train_y[train_y.columns.values[1]].max(), 10)
        xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
        # print(3)
        try:
            Z = xx_pred * res.params[0] + yy_pred * res.params[1]

            ax.scatter(train_y[train_y.columns.values[0]].values.reshape(-1, 1),
                       train_y[train_y.columns.values[1]].values.reshape(-1, 1), train_x, color='k', zorder=15,
                       marker='o', alpha=0.5)
            ax.plot_surface(xx_pred, yy_pred, Z, rstride=1, cstride=1, alpha=0.4)
            ax.set_xlabel(train_y.columns.values[0])
            ax.set_ylabel(train_y.columns.values[1])
            ax.set_zlabel(train_x.columns.values[0])

            # plt.show()
            fname = str(uuid.uuid4()) + ".jpg"

            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'Results/linearRegression/')

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            plt.savefig(results_dir + fname, bbox_inches='tight')
        except Exception as e:
            print(e)

    res2 = res.summary().tables[2].as_html()
    res1 = res.summary().tables[1].as_html()
    res0 = res.summary().tables[0].as_html()
    with open(results_dir + fname, 'rb') as img:
        base64_string = base64.b64encode(img.read())

    base64_string = urllib.parse.quote(base64_string)
    return {'summary': res1 + res0 + res2, 'fname': base64_string}


def logisticRegression(train_x, train_y):
    scaler = StandardScaler()
    print(1)
    train_x = scaler.fit_transform(train_y.values.reshape(-1, 1))
    model = LogisticRegression()
    model.fit(train_x, train_y)
    print(2)
    try:
        plt.figure(1, figsize=(4, 3))
        plt.clf()
    except Exception as e:
        print("================")
        print(e)
    print(3)
    plt.scatter(train_x.ravel(), train_y.values.ravel(), color="black", zorder=20)
    print(4)
    X_test = np.linspace(-5, 10, 300)

    try:
        loss = expit(X_test * model.coef_ + model.intercept_).ravel()
        plt.plot(X_test, loss, color="red", linewidth=3)
        plt.legend(
            ("Logistic Regression Model", "Linear Regression Model"),
            loc="lower right",
            fontsize="small",
        )

    except Exception as e:
        print(e)

    fname = str(uuid.uuid4()) + ".jpg"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/LogisticRegression/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.savefig(results_dir + fname, bbox_inches='tight')
    with open(results_dir + fname, 'rb') as img:
        base64_string = base64.b64encode(img.read())

    base64_string = urllib.parse.quote(base64_string)
    return {"coef": str(model.coef_), "fname": base64_string}


def DecisionTree(train_x, train_y, depth):
    train_x = train_x.astype(float)
    train_y = train_y.astype(float)
    tree_model = DecisionTreeClassifier(max_depth=int(depth))

    tree_model.fit(train_x, train_y)
    r = export_text(tree_model)

    fig = plt.figure(figsize=(10, 10))

    _ = tree.plot_tree(tree_model,
                       # feature_names=train_x.columns.values,
                       # class_names=train_y.columns.values,
                       feature_names=train_x.columns.values,
                       class_names=train_y.columns.values,
                       filled=True)

    fname = str(uuid.uuid4()) + ".jpg"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/linearRegression/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.savefig(results_dir + fname)

    with open(results_dir + fname, 'rb') as img:
        base64_string = base64.b64encode(img.read())

    base64_string = urllib.parse.quote(base64_string)
    return {'fname': base64_string}


def KNN(train_x, train_y, test_x, cluster):
    knn = KNeighborsClassifier(n_neighbors=3)
    print(train_x)
    print(train_y)
    knn.fit(train_x, train_y.values.ravel())
    print(1)
    test_x = pd.DataFrame([test_x], columns=train_x.columns)
    print(2)
    test_y = knn.predict(test_x)
    print(3)
    # fig = plt.figure(1)
    #
    # plt.scatter(train_x.iloc[ :, 0], train_x.iloc[ :, 1], c=train_y.values)
    # plt.scatter(test_x.iloc[ :, 0], test_x.iloc[ :, 1], c=test_y)
    # plt.annotate("predict", xy=test_x)

    X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=30)

    blue = X[y == 0]
    red = X[y == 1]

    # 랜덤한 새로운 점 생성
    newcomer = np.random.randn(1, 2)
    fig = plt.figure(1)

    plt.scatter(train_x.values[:, 0], train_x.values[:, 1], 80, 'r', '^')
    plt.scatter(test_x.values[:, 0], test_x.values[:, 1], 80, 'g', 'o')
    #
    # # n_neighbors=3
    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(train_x, train_y.values.ravel())
    # pred = knn.predict(test_x)
    #
    # # 표기
    # plt.annotate('predict', xy=test_x[0], xytext=(test_x[0]), fontsize=12)

    fname = str(uuid.uuid4()) + ".jpg"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/KNN/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.savefig(results_dir + fname, bbox_inches='tight')

    with open(results_dir + fname, 'rb') as img:
        base64_string = base64.b64encode(img.read())

    base64_string = urllib.parse.quote(base64_string)
    return {'fname': base64_string}


def svm(train_x, train_y, test_x, kernel):
    if kernel == "BDI001":
        kernel = "linear"
    elif kernel == "BDI002":
        kernel = "rbf"
    elif kernel == "BDI003":
        kernel = "poly"

    # rbf, poly => 비선형, linear => 선형)
    clf = SVC(kernel=kernel)
    clf.fit(train_x, train_y.values.ravel())
    print(23)
    # test_x = pd.DataFrame([test_x], columns=train_x.columns)

    if kernel == "linear":
        y_predict = clf.predict(train_x)  # y_predict 값에 x_test를 통해 예측한 값이 할당됩니다.
        print(1)
        plt.plot(train_x[train_x.columns[0]], train_x[train_x.columns[1]])
        print(2)
        try:
            plot_svc_decision_boundary(clf, 0, 6)
        except Exception as e:
            print(e)
    else:
        plot_predictions(clf, [-1.5, 2.5, -1, 1.5])
        plt.plot(train_x[train_x.columns[0]], train_x[train_x.columns[1]])
    # accuracy_score(train_x, y_predict)  # accuracy_score을 통해 단순히 몇개를 맞추었는지 알 수 있습니다.
    #    plt.scatter(train_x.iloc[:, 0], train_x.iloc[:, 1], cmap=plt.cm.coolwarm)
    #
    #    ax = plt.gca()
    #    xlim = ax.get_xlim()
    #    ylim = ax.get_ylim()
    #    try:
    #        xx = np.linspace(xlim[0], xlim[1], 2)
    #        yy = np.linspace(ylim[0], ylim[1], 3)
    #        YY, XX = np.meshgrid(yy, xx)
    #        xy = np.vstack([XX.ravel(), YY.ravel()]).T
    #        print(XX)
    #    except Exception as e:
    #        print(e)
    #
    #    Z = clf.decision_function(train_x).reshape(XX.shape)
    #    print(XX)
    #    print(YY)
    #    print(Z)
    #    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    #    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=60, facecolors='r')
    #    print(2)
    # plt.show()
    fname = str(uuid.uuid4()) + ".jpg"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/SVM/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.savefig(results_dir + fname, bbox_inches='tight')

    with open(results_dir + fname, 'rb') as img:
        base64_string = base64.b64encode(img.read())

    base64_string = urllib.parse.quote(base64_string)

    return {'classfication': classification_report(train_y, y_predict), 'fname': base64_string}


def naive_bayes(train_x, train_y, test_x, test_y):
    nb = GaussianNB()
    nb.fit(train_x, train_y)

    predict = nb.predict(test_x)

    print("Number of mislabeled points out of a total %d proints : %d" % (test_x.shape[0], (test_y != predict).sum()))


def randomForest(train_x, train_y, estimators, depth):
    clf = RandomForestClassifier(n_estimators=int(estimators), max_depth=int(depth), random_state=0)
    clf.fit(train_x, train_y)

    # predict1 = clf.predict(test_x)
    # print(accuracy_score(test_y, predict1))

    fpath = [0 for i in range(len(clf))]

    for i in range((len(clf))):
        print(i)
        fig = plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(clf[i],
                           feature_names=train_x.columns.values,
                           class_names=train_y.columns.values,
                           filled=True)

        fname = str(uuid.uuid4()) + ".jpg"

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Results/RandomForest/')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(results_dir + fname, bbox_inches='tight')

        with open(results_dir + fname, 'rb') as img:
            base64_string = base64.b64encode(img.read())

        fpath[i] = urllib.parse.quote(base64_string)

    return ({'fname': fpath})


def kMeans(train_x, train_y):
    print(train_x)
    #    train_x["TU_0007_0000000002.SCHL_THEME"] = train_x["TU_0007_0000000002.SCHL_THEME"].str.extract('(\d+)')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(train_x)
    print(3)
    # 결과 확인
    result_by_sklearn = train_x.copy()
    print(4)
    result_by_sklearn["cluster"] = kmeans.labels_
    print(kmeans.labels_)
    scatter = sns.scatterplot(train_x.columns[0], train_x.columns[1], hue=train_y[train_y.columns[0]],
                              data=result_by_sklearn, palette="Set2")
    plt.legend(loc='best')
    fname = str(uuid.uuid4()) + ".jpg"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/KMeans/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.savefig(results_dir + fname, bbox_inches='tight')

    with open(results_dir + fname, 'rb') as img:
        base64_string = base64.b64encode(img.read())

    base64_string = urllib.parse.quote(base64_string)

    return {"fname": base64_string}


def min_max_normalize(lst):
    normalized = []

    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)

    return normalized


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    print(3)
    x0 = np.linespace(xmin, xmax, 200)
    print(4)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    print(5)
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, 'k-', linewidth=2)
    plt.plot(x0, gutter_up, 'k--', linewidth=2)
    plt.plot(x0, gutter_down, 'k--', linewieth=2)


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 3000)
    x1s = np.linspace(axes[0], axes[3], 3000)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap=plt.cm.brg, alpah=0.2)
    plt.contour(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


def histogram(df, statX):
    fig = go.Figure()

    for i in df.columns:
        try:
            df[i] = train_x[i].apply(pd.to_numeric)
        except Exception as e:
            print(e)
        fig.add_trace(go.Histogram(x=df[i]))

    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)

    fname = str(uuid.uuid4()) + ".html"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/Histogram/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    fig.write_html(results_dir + fname)

    return {"fname": fname}


def bar(df, statX, statY, statColor):
    print("===========bar")
    # df = df.loc[:, ~df.T.duplicated()]

    for i in df[statX.split(",") + statY.split(",")].columns:
        df[i] = df[i].apply(pd.to_numeric)

    if len(statY.split(",")) >= 2:
        fig = go.Figure()
        for i in range(len(statY)):
            fig.add_trace(go.Bar(
                df,
                x=statX[i],
                y=statY[i],
                name=statY[i]
            ))

        fig.update_layout(barmode='group')
    else:
        try:
            fig = px.bar(df, x=statX, y=statY, color=statColor)
        except:
            fig = px.bar(df, x=statX, y=statY)

    # try:
    #     for i in train_y.columns:
    #         try:
    #             train_y[i] = train_y[i].apply(pd.to_numeric)
    #         except Exception as e:
    #             print(e)
    #
    #         train_y = train_y.loc[:, ~train_y.T.duplicated()]
    #     print("=========================")
    #     print(train_y.columns)
    #     if len(train_y.columns) >= 2:
    #         print("====================")
    #         fig = go.Figure()
    #         fig.add_trace(go.Bar(
    #             x=train_y[train_y.columns[0]],
    #             y=train_y[train_y.columns[1]],
    #             name=train_y.columns[1]
    #         ))
    #         fig.add_trace(go.Bar(
    #             x=train_x[train_x.columns[0]],
    #             y=train_x[train_x.columns[1]],
    #             name=train_x.columns[1]
    #         ))
    #
    #         fig.update_layout(barmode='group')
    #
    #     else:
    #         try:
    #             fig = px.bar(x=train_x[train_x.columns[0]], y=train_x[train_x.columns[1]], color=train_y[train_y.columns[0]])
    #         except Exception as e:
    #             print(e)
    # except:
    #     fig = px.bar(x = train_x[train_x.columns[0]], y = train_x[train_x.columns[1]])

    fname = str(uuid.uuid4()) + ".html"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/Bar/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    fig.write_html(results_dir + fname)

    return {"fname": fname}


def line(df, statX, statY, statColor):
    fig = go.Figure()
    try:
        fig = px.line(df, x=statX, y=statY, color=statColor)
    except Exception as e:
        fig = px.line(df, x=statX, y=statY)
    print(df)
    # fig = px.line( x = train_x[train_x.columns[0]], y = train_x[train_x.columns[1]], color=train_y[train_y.columns[0]])

    fname = str(uuid.uuid4()) + ".html"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/Line/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    fig.write_html(results_dir + fname)

    return {"fname": fname}


def scatter(df, statX, statY, statZ, statColor, statSize, statSymbol):
    fig = go.Figure()
    df = df.loc[:, ~df.T.duplicated()]
    # xlen = len(train_x.columns)
    # ylen = 0
    # try:
    #     ylen = len(train_y.columns)
    # except Exception as e:
    #     print(ylen)
    #
    # if xlen == 2 and ylen == 2:
    #     fig = px.scatter(x=train_x[train_x.columns[0]], y = train_x[train_x.columns[1]], size= train_y[train_y.columns[0]], symbol=train_y[train_y.columns[1]])
    #
    # elif xlen == 2 and ylen == 1:
    #     fig = px.scatter(x=train_x[train_x.columns[0]], y = train_x[train_x.columns[1]], color= train_y[train_y.columns[0]], size=train_y[train_y.columns[0]])
    # elif xlen == 3 and ylen == 1:
    #     fig = px.scatter_3d(x = train_x[train_x.columns[0]], y = train_x[train_x.columns[1]], z = train_x[train_x.columns[2]], size = train_y[train_y.columns[0]])
    # elif xlen == 3:
    #     fig = px.scatter_3d(train_x, x = train_x.columns[0], y = train_x.columns[1], z = train_x.columns[2])
    #
    # else:
    #     fig = px.scatter(x=train_x[train_x.columns[0]], y = train_x[train_x.columns[1]])
    #
    try:
        print("========" + statSymbol)
        fig = px.scatter_3d(df, x=statX, y=statY, z=statZ, color=statColor, size=statSize, symbol=statSymbol)
    except:
        print(statSymbol)
        fig = px.scatter(df, x=statX, y=statY, color=statColor, size=statSize, symbol=statSymbol)

    fname = str(uuid.uuid4()) + ".html"

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/Scatter/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    fig.write_html(results_dir + fname)

    return {"fname": fname}


