import json
from pymongo import MongoClient
from django.http import HttpResponse

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
        #database내에 있는 Collection목록을 가지옴
        collectionList = database.list_collection_names()

        #Collection 존재 여부 체크
        if collectionNm in collectionList:
            return HttpResponse(json.dumps({'DB_YN':'Y'}), 200)
        else:
            newcol = database.create_collection(collectionNm)
            print(database.list_collection_names())
            return HttpResponse(json.dumps({'DB_YN':'N'}), 200)
    except:
        resp = HttpResponse(status=400)
        return resp