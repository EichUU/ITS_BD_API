import json

import bson
import pql
import pandas as pd
from pymongo import MongoClient
from django.http import HttpResponse
from bson.json_util import dumps, loads

from bson.objectid import ObjectId

#로컬
#mongo_host = "192.168.0.193"

#도커
mongo_host = "localhost"
mongo_port = 27017


# 몽고디비 collection 생성
# host : mongodb 주소
# port : mongodb 포트
# dbs : mongodb database
# collection : mongodb collection
def createCollection(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        dbs = body['db']
        collection = body['collection']
    else:
        collection = request.GET.get("collection")
        dbs = request.GET.get("dbs")

    mongoconn = MongoClient(mongo_host, int(mongo_port))
    database = mongoconn[dbs]
    try:
        # database내에 있는 Collection목록을 가지옴
        collectionList = database.list_collection_names()
        # Collection 존재 여부 체크
        if collection in collectionList:
            return HttpResponse(json.dumps({'DB_YN': 'N'}), 200)
        else:
            newcol = database.create_collection(collection)
            print(database.list_collection_names())
            return HttpResponse(json.dumps({'DB_YN': 'Y'}), 200)
    except:
        resp = HttpResponse(status=400)
        return resp


# 몽고디비 저장!
# db : mongodb database
# col : mongodb collection
# path : 저장된 파일의 경로
def documentInsert(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        db = body['db']
        col = body['col']
        path = body['path']
        # header에 들어갈 컬럼들을 string으로 받아 json으로 변경함
        header = json.loads(body['header'])
    else:
        db = request.GET.get("db")
        col = request.GET.get("col")
        path = request.GET.get("path")
        # header에 들어갈 컬럼들을 string으로 받아 json으로 변경함
        header = json.loads(request.GET.get("header"))

    print("db====>"+db)
    print("db====>" + col)
    print("db====>" + path)
    # documentKey값을 배열에 저장
    documentKey = []

    print(header)
    for docuKey in list(header):
        # DOCUMENT_KEY에서 '.'이 있는 경우 . 뒤에 있는 항목으로 저장
        if docuKey["DOCUMENT_KEY"].find(".") > 0:
            documentKey.append(docuKey["DOCUMENT_KEY"].split('.')[1])
        else:
            documentKey.append(docuKey["DOCUMENT_KEY"])

    mongo = MongoClient(mongo_host, int(mongo_port))
    try:
        df = pd.read_excel(path, header=1, index_col=False, names=documentKey, skiprows=[0])
        df = df.astype(str)
        df.reset_index(inplace=True)

        df = df.drop(columns=["index"])
        dic = df.to_dict("records")
        database = mongo[db]
        collection = database[col]
    except Exception as e:
        print(e)

    try:
        collection.insert_many(dic)
        resp = HttpResponse("{'SUCCESS_YN':'Y'}")
        resp.status_code = 200

        return resp
    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp


# 몽고디비 select
# db : mongodb database
# col : mongodb collection
# query : mongodb select 조건문(json 형태로 입력)

# pql쿼리는 SUST_CD=="A-1" 요런식으로 사용
# pql 사용 법 : https://github.com/alonho/pql 참고

##############################################################################################################
# pql 처음 설치 시 (3.8버전 이상)                                                                               #
# External Libraries -> site-packages -> pql -> matching.py                                                 #
# https://github.com/comfuture/pql/blob/331739a02ccde5a68674e80e20181f687abd13a2/pql/matching.py 으로 수정    #
##############################################################################################################
def mongoSelect(request):
    body_unicode = request.body.decode('utf-8')

    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        db = body['db']
        col = body['col']
        query = body['query']
        #limit = body['limit']
        #sort = body['sort']
        #skip = body['skip']

    else:
        db = request.GET.get("db")
        col = request.GET.get("col")
        query = request.GET.get("query")
        #limit = request.GET.get("limit")
        #sort = request.GET.get("sort")
        #skip = request.GET.get("skip")

    #if limit is None:
    #    limit = 1

    #if sort is None:
    #    sort = [("_id", -1)]

    #if skip is None:
    #    skip = 0

    mongo = MongoClient(mongo_host, int(mongo_port))
    database = mongo[db]
    collection = database[col]

    try:
        if query is None or query == '':
            cur = objectIdDecoder(collection.find({}))

            #.sort(sort).limit(limit))
        else:
            cur = objectIdDecoder(collection.find(pql.find(query)))

        list_cur = list(cur)
        cur = dumps(list_cur)
        resp = HttpResponse(cur)
        resp.status_code = 200
        return resp

    except Exception as e:
        print(e)
        resp = HttpResponse(status=400)
        return resp


# 몽고디비 삭제가능여부 체크
# database는 collection이 존재하면 삭제 불가, collection은 document가 존재하면 삭제 불가
# db : mongodb database
# col : mongodb collection
# query : mongodb select 조건문(json 형태로 입력)
def deleteChk(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        db = body['db']
        col = body['col']

    else:
        db = request.GET.get("db")
        col = request.GET.get("col")

#로컬
#    host = "localhost"
#    port = 27017

#도커
    host = "172.17.0.6"
    port = 20000 #샤딩

    mongo = MongoClient(host, int(port))
    database = mongo[db]

    try:
        if col == "":
            collection = database.list_collection_names()
            if len(collection) == 0:
                resp = HttpResponse("{'deleteYn':'Y'}")
                resp.status_code = 200
            else:
                resp = HttpResponse("{'deleteYn':'N'}")
                resp.status_code = 200
        else:
            collection = database[col]
            cur = collection.find_one({})
            print(len(list(cur)))
            if len(list(cur)) == 0:
                resp = HttpResponse("{'deleteYn':'Y'}")
                resp.status_code = 200
            else:
                print(len(list(cur)))
                resp = HttpResponse("{'deleteYn':'N'}")
                resp.status_code = 200
        return resp
    except Exception as e:
        resp = HttpResponse(status=400)
        return resp


# 성과지표 몽고디비 collection 생성
# host : mongodb 주소
# port : mongodb 포트
# dbs : mongodb database
# collection : mongodb collection
def createIRCollection(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        collectionNm = body['collection']
    else:
        collectionNm = request.GET.get("collection")

    dbs = "IR_RESULT"
    mongoconn = MongoClient(mongo_host, int(mongo_port))
    database = mongoconn[dbs]
    try:
        # database내에 있는 Collection목록을 가지옴
        collectionList = database.list_collection_names()

        # Collection 존재 여부 체크
        if collectionNm in collectionList:
            return HttpResponse(json.dumps({'DB_YN': 'Y'}), 200)
        else:
            newcol = database.create_collection(collectionNm)
            print(database.list_collection_names())
            return HttpResponse(json.dumps({'DB_YN': 'N'}), 200)
    except:
        resp = HttpResponse(status=400)
        return resp


def objectIdDecoder(list):
    results = []
    for document in list:
        document['_id'] = str(document['_id'])
        results.append(document)
    return results
