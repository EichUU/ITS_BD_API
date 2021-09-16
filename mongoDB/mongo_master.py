import json
import pandas as pd
from pymongo import MongoClient
from django.http import HttpResponse
from bson.json_util import dumps, loads

mongo_host = "192.168.0.100"
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
            return HttpResponse(json.dumps({'DB_YN':'Y'}), 200)
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

    else:
        db = request.GET.get("db")
        col = request.GET.get("col")
        path = request.GET.get("path")

    mongo = MongoClient(mongo_host, int(mongo_port))
    df = pd.read_excel(path)
    df.reset_index(inplace=True)
    dic = df.to_dict("records")
    database = mongo[db]
    collection = database[col]
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
def mongoSelect(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        db = body['db']
        col = body['col']
        query = body['query']

    else:
        db = request.GET.get("db")
        col = request.GET.get("col")
        query = request.GET.get("query")

    host = "localhost"
    port = 27017
    mongo = MongoClient(host, int(port))
    database = mongo[db]
    collection = database[col]
    try:
        if query is None or query == '':
            cur = collection.find({}).sort([("_id", -1)])
        else:
            cur = collection.find(json.loads(query)).sort([("_id", -1)])

        cur = dumps(list(cur))
        resp = HttpResponse(cur)
        resp.status_code = 200
        return resp

    except Exception as e:
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

    host = "localhost"
    port = 27017
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