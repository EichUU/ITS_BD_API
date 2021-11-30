import json
import pyodbc
import os

from django.shortcuts import render
from django.http import HttpResponse
from pyhive import hive
from dbcon.connDB import get_conn

# Create your views here.
def get_table(request):

    # UPMS(티베로), BIG_DATA(하이브)
    # POST로 받은 값이 null이면 GET으로 받음
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        tspace = body['tspace']
        tcate = body['tcate']

    else:
        tspace = request.GET.get("tspace")
        tcate = request.GET.get("tcate")

    # get_conn(dbtype, dsname, dbuser, dbpwd, dbhost, dbport):
    # dbhost, dbport is not used in tibero. because tibero is connected of odbc
    # hive don't use dbuser, dbpwd
    if tcate == "BIG_DATA":
        cur = get_conn("hive", "", "", "", "192.168.0.133", 10000)
        cur.execute("show tables")

    elif tcate == "UPMS":
        cur = get_conn("tibero", "TIBERO", "UPMS", "UPMS12#$", "", "")

        cur = cur.execute(
            "select A.TABLE_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, A.NUM_ROWS NUM_ROWS from ALL_TABLES A, ALL_TAB_COMMENTS B where A.TABLE_NAME = B.TABLE_NAME AND A.OWNER = '" + tspace + "' " +
            "UNION ALL SELECT A.VIEW_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, 0 NUM_ROWS from ALL_VIEWS A, USER_TAB_COMMENTS B where A.VIEW_NAME = B.TABLE_NAME AND A.OWNER='" + tspace + "' ORDER BY TABLE_NAME")

    elif tcate == "SQL":
        cur = get_conn("tibero", "TIBERO", "UPMS", "UPMS12#$", "", "")
        #스케쥴링에 등록 된 동적SQL에 대한 TABLE정보를 가지고 옴
        cur = cur.execute(
            "select A.TABLE_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, A.NUM_ROWS NUM_ROWS "
            "from ALL_TABLES A, ALL_TAB_COMMENTS B where A.TABLE_NAME = B.TABLE_NAME AND A.OWNER = 'UPMS' " +
            "AND A.TABLE_NAME IN (select OBJ_TABLE from TPSYB060_M where MTHD_GB = 'SB4003') ORDER BY A.TABLE_NAME")

    elif tcate == "NoSql":
        cur = get_conn("tibero", "TIBERO", "UPMS", "UPMS12#$", "", "")
        #비정형계(NoSql:MongoDB) IR_RESULT유저를 제외함.
        cur = cur.execute(
            "SELECT DATABASE_NM TABLE_NAME, UP_DATABASE_NM TABLESPACE_NAME, DATABASE_COMMENT COMMENTS, 0 NUM_ROWS, DATABASE_ID ID_KEY "
            "FROM T_MONGO_M  " +
            "WHERE DATABASE_LEVEL='2' AND NVL(USE_YN, 'Y') = 'Y' AND UP_DATABASE_ID NOT IN ('00001') ORDER BY UP_DATABASE_NM, DATABASE_NM ")

    else:
        return HttpResponse("No tspace")

    array_dict = []
    for it in cur.fetchall():
        tables = dict()
        tables["TABLE_NAME"] = it[0]
        try:
            tables["TABLESPACE_NAME"] = it[1]
        except IndexError:
            tables["TABLESPACE_NAME"] = tspace

        try:
            tables["COMMENTS"] = it[2]
        except IndexError:
            tables["COMMENTS"] = ''

        try:
            tables["NUM_ROWS"] = it[3]
        except IndexError:
            tables["NUM_ROWS"] = ''

        try:
            tables["ID_KEY"] = it[4]
        except IndexError:
            tables["ID_KEY"] = ''

        array_dict.append(tables)
        json_array = json.dumps(array_dict)

    cur.close()
    return HttpResponse(json_array, content_type="application/json; charset=utf-8")


def get_column(request):
    tspace = request.args.get("tspace")
    tname = request.args.get("tname")
    host = "192.168.0.133"
    port = 10000
    conn = hive.Connection(host=host, port=port, auth="NOSASL", database="default")
    cur = conn.cursor()
    query = "describe " + tname
    cur.execute(query)
    array_dict = []
    col_seq = 10
    for it in cur.fetchall():
        print(it)
        columns = dict()
        columns["Cname"] = it[0]
        columns["Tname"] = tname
        columns["ColType"] = it[1]
        columns["ColSeq"] = col_seq
        array_dict.append(columns)
        col_seq += 10
    json_array = json.dumps(array_dict)
    return str(json_array), 200


def insertTable(request):
    #Table정보를 저장한다.

    #tspace = request.POST.get("tspace")
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        strTableSpaceNm = body['tableSpace']
        strTableNm = body['tableNm']
        strTableComment = body['tableComment']
        strCategoryNm = body['cateNm']
        strUserId = body['userID']
        strUserIp = body['userIP']
    else:
        strTableSpaceNm = request.GET.get("tableSpace")
        strTableNm = request.GET.get('tableNm')
        strTableComment = request.GET.get('tableComment')
        strCategoryNm = request.GET.get('cateNm')
        strUserId = request.GET.get('userID')
        strUserIp = request.GET.get('userIP')

    print(strTableNm)
    print(strTableComment)
    #티베로DB > tablespace에  tablename 존재 유무 검사
    conn = pyodbc.connect('DSN=TIBERO;UID=UPMS;PWD=UPMS12#$')
    conn.setencoding(encoding='utf-8')
    strSql = """SELECT MDS_TABLE_SEQ FROM T_MDS_TABLES_M WHERE MDS_TABLE_USER='""" + strTableSpaceNm +"""' AND MDS_TNAME='"""+ strTableNm +"""'"""
    cur = conn.execute(strSql)
    rowData = cur.fetchall()
    conn.close()

    array_dict = []
    for row in rowData:
        tables = dict()
        tables["status"] = "failed"
        tables["msg"] = "table exist"
        array_dict.append(tables)
        json_array = json.dumps(array_dict)
        return HttpResponse(json_array, content_type="application/json; charset=utf-8")

    if strCategoryNm == "BIG_DATA":
        # Hive
        host = "192.168.0.133"
        port = 10000
        conn = hive.Connection(host=host, port=port, auth="NOSASL", database="default")
    elif strCategoryNm == "UPMS":
        # 티베로
        DSNNAME = 'TIBERO'
        DBUSER = 'UPMS'
        DBPWD = 'UPMS12#$'
        conn = pyodbc.connect('DSN=' + DSNNAME + ';UID=' + DBUSER + ';PWD=' + DBPWD)
        #해당 Table에 대한 컬럼 정보 및 pk정보를 가지고 옴
        rowColumnInfo = getTableColumnPK(conn, DBUSER, strTableNm)
    elif strCategoryNm == "SQL":
        # 동적SQL문
        DSNNAME = 'TIBERO'
        DBUSER = 'UPMS'
        DBPWD = 'UPMS12#$'
        conn = pyodbc.connect('DSN=' + DSNNAME + ';UID=' + DBUSER + ';PWD=' + DBPWD)
        #해당 Table에 대한 컬럼 정보 및 pk정보를 가지고 옴
        rowColumnInfo = getTableColumnPK(conn, DBUSER, strTableNm)
    elif strCategoryNm == "NoSql":
        # NoSql(MongoDB)에 관한 Collection정보를 가져옴
        DSNNAME = 'TIBERO'
        DBUSER = 'UPMS'
        DBPWD = 'UPMS12#$'
        conn = pyodbc.connect('DSN=' + DSNNAME + ';UID=' + DBUSER + ';PWD=' + DBPWD)
        #해당 Table에 대한 컬럼 정보 및 pk정보를 가지고 옴
        rowColumnInfo = getNoSqlDocument(conn, strTableSpaceNm, strTableNm)
    else:
        return HttpResponse("No tspace")


    conn.close()

    #MDS관리 테이블에 정보를 저장
    mdsConn = pyodbc.connect('DSN=TIBERO;UID=UPMS;PWD=UPMS12#$')

    #서버가 원도용가 아닐경우 아래 주석을 풀어주기
    mdsCursor = mdsConn.cursor()
    mdsCursor.execute("""INSERT INTO T_MDS_TABLES_M (MDS_TABLE_SEQ, MDS_CATE_NM, MDS_TABLE_USER, MDS_TNAME, MDS_TABLE_DESCRIPTION, MAKE_ID, INPT_ID, INPT_DT, INPT_IP, UPDT_ID, UPDT_DT, UPDT_IP) 
                        VALUES (SEQ_T_MDS_TABLES_01.NEXTVAL,?,?,?,?,?,?,SYSDATE,?,?,SYSDATE,?)""",
                        strCategoryNm, strTableSpaceNm, strTableNm, strTableComment, strUserId, strUserId, strUserIp, strUserId, strUserIp)
    mdsConn.commit()


    if strCategoryNm == "UPMS" or strCategoryNm == "SQL" or strCategoryNm == "NoSql":
        for it in rowColumnInfo:
            mdsCursor.execute("""INSERT INTO T_MDS_COLUMNS_D (MDS_COLUMN_SEQ, MDS_TABLE_SEQ, MDS_COLUMN_NAME, MDS_COLUMN_TYPE, MDS_COLUMN_DESCRIPTION, COLUMN_KEY_YN, COLUMN_ID_SEQ, MAKE_ID, INPT_ID, INPT_DT, INPT_IP, UPDT_ID, UPDT_DT, UPDT_IP) 
                                            VALUES (?, SEQ_T_MDS_TABLES_01.CURRVAL, ?, ?, ?, ?, ?, ?, ?, SYSDATE, ?, ?, SYSDATE, ?)""",
                              int(it[4]), it[0], it[1], it[2], it[3].strip(), int(it[4]), strUserId, strUserId, strUserIp, strUserId, strUserIp)
            mdsConn.commit()
    # Connection 닫기
    mdsConn.close()

    #결과값을 JSON으로 리턴
    tables = dict()
    tables["status"] = "ok"
    tables["msg"] = "table insert."
    array_dict.append(tables)
    json_array = json.dumps(array_dict)
    return HttpResponse(json_array,  content_type="application/json; charset=utf-8")

#컬럼 정보 및 PK정보를 가지고 옴
def getTableColumnPK( dbConn, tableOwner, tableNm ) :
    dbConn.setencoding(encoding='utf-8')
    strSql = """SELECT T2.COLUMN_NAME, T1.DATA_TYPE, T2.COMMENTS
                     , DECODE(NVL(T3.CONSTRAINT_TYPE, 'N'), 'N', 'N', 'Y') AS PK_YN, T1.COLUMN_ID
                FROM ALL_TAB_COLUMNS T1 
                   , ALL_COL_COMMENTS T2 
                   , ( SELECT A.TABLE_NAME, B.COLUMN_NAME, A.CONSTRAINT_NAME, A.CONSTRAINT_TYPE, A.SEARCH_CONDITION
                       FROM USER_CONSTRAINTS A, USER_CONS_COLUMNS B
                       WHERE A.CONSTRAINT_NAME = B.CONSTRAINT_NAME
                         AND A.OWNER='""" + tableOwner +"""' AND A.TABLE_NAME ='"""+ tableNm +"""' AND A.CONSTRAINT_TYPE = 'P'
                     ) T3
                WHERE T1.OWNER = T2.OWNER
                  AND T1.TABLE_NAME = T2.TABLE_NAME
                  AND T1.COLUMN_NAME = T2.COLUMN_NAME
                  AND T1.TABLE_NAME = T3.TABLE_NAME(+)
                  AND T1.COLUMN_NAME = T3.COLUMN_NAME(+)
                  AND T1.OWNER='""" + tableOwner +"""' AND T1.TABLE_NAME ='"""+ tableNm +"""'
                ORDER BY T1.COLUMN_ID"""
    cur = dbConn.execute(strSql)
    retRow = cur.fetchall()

    return retRow

#NoSql Document Key, Key 설명 정보를 가지고 옴
def getNoSqlDocument( dbConn, tableOwner, tableNm ) :
    print("==" + tableOwner)
    dbConn.setencoding(encoding='utf-8')
    strSql = """SELECT T1.DOCUMENT_KEY AS COLUMN_NAME, 'VARCHAR' AS DATA_TYPE, T1.DOCUMENT_COMMENT AS COMMENTS 
                     , 'N' AS PK_YN, T1.SORT_ORD AS COLUMN_ID
                FROM T_MONGO_DOCU_D T1 
                   , T_MONGO_M T2 
                WHERE T1.DATABASE_ID = T2.DATABASE_ID
                  AND NVL(T1.USE_YN, 'Y') = 'Y'
                  AND T2.UP_DATABASE_NM='""" + tableOwner +"""' AND T2.DATABASE_NM ='"""+ tableNm +"""'
                ORDER BY T1.SORT_ORD"""
    cur = dbConn.execute(strSql)
    retRow = cur.fetchall()
    print(retRow)

    return retRow