import json
import pyodbc
import cx_Oracle
import os

from django.shortcuts import render
from django.http import HttpResponse
from pyhive import hive
from dbcon.connDB import get_conn


# Create your views here.
def get_table(request):
    json_array = ""
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
    print(tspace, tcate)
    # get_conn(dbtype, dsname, dbuser, dbpwd, dbhost, dbport):
    # dbhost, dbport is not used in tibero. because tibero is connected of odbc
    # hive don't use dbuser, dbpwd
    if tcate == "BIG_DATA":
        try:
            print(1)
            #            cur = get_conn("hive", "", "", "", "172.17.0.2", 10000)
            cur = get_conn("mariadb", "hive", "root", "mapsco1!", "mapsco.kr", "3307")
            print(2)
            cur.execute(
                """SELECT T1.TBL_NAME AS TABLE_NAME, 'BIG_DATA' AS TABLESPACE_NAME, T2.PARAM_VALUE AS COMMENTS FROM TBLS T1 LEFT OUTER JOIN TABLE_PARAMS T2 ON T1.TBL_ID = T2.TBL_ID AND T2.PARAM_KEY = 'COMMENT'""")
        except Exception as e:
            print(e)

    elif tcate == "UPMS":
        cur = get_conn("tibero", "TIBERO", "UPMS", "UPMS12#$", "", "")

        cur = cur.execute(
            "select A.TABLE_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, A.NUM_ROWS NUM_ROWS from ALL_TABLES A, ALL_TAB_COMMENTS B where A.TABLE_NAME = B.TABLE_NAME AND A.OWNER = '" + tspace + "' " +
            "UNION ALL SELECT A.VIEW_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, 0 NUM_ROWS from ALL_VIEWS A, USER_TAB_COMMENTS B where A.VIEW_NAME = B.TABLE_NAME AND A.OWNER='" + tspace + "' ORDER BY TABLE_NAME")

    elif tcate == "SQL":
        cur = get_conn("tibero", "TIBERO", "UPMS", "UPMS12#$", "", "")
        # 스케쥴링에 등록 된 동적SQL에 대한 TABLE정보를 가지고 옴
        cur = cur.execute(
            "select A.TABLE_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, A.NUM_ROWS NUM_ROWS "
            "from ALL_TABLES A, ALL_TAB_COMMENTS B where A.TABLE_NAME = B.TABLE_NAME AND A.OWNER = 'UPMS' " +
            "AND A.TABLE_NAME IN (select OBJ_TABLE from TPSYB060_M where MTHD_GB = 'SB4003') ORDER BY A.TABLE_NAME")

    elif tcate == "NoSql":
        cur = get_conn("tibero", "TIBERO", "UPMS", "UPMS12#$", "", "")
        # 비정형계(NoSql:MongoDB) IR_RESULT유저를 제외함.
        cur = cur.execute(
            "SELECT DATABASE_NM TABLE_NAME, UP_DATABASE_NM TABLESPACE_NAME, DATABASE_COMMENT COMMENTS, 0 NUM_ROWS, DATABASE_ID ID_KEY "
            "FROM T_MONGO_M  " +
            "WHERE DATABASE_LEVEL='2' AND NVL(USE_YN, 'Y') = 'Y' AND UP_DATABASE_ID NOT IN ('00001') ORDER BY UP_DATABASE_NM, DATABASE_NM ")

    elif tcate == "BDAS":
        cur = get_conn("oracle", "XE", "BDAS", "BDAS12#$", "mapsco.kr:1531/XE", "")
        cur = cur.execute(
            "select A.TABLE_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, A.NUM_ROWS NUM_ROWS from ALL_TABLES A, ALL_TAB_COMMENTS B where A.TABLE_NAME = B.TABLE_NAME AND A.OWNER = '" + tspace + "' " +
            "UNION ALL SELECT A.VIEW_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, 0 NUM_ROWS from ALL_VIEWS A, USER_TAB_COMMENTS B where A.VIEW_NAME = B.TABLE_NAME AND A.OWNER='" + tspace + "' ORDER BY TABLE_NAME")

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
    # Table정보를 저장한다.

    # tspace = request.POST.get("tspace")
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

    # 티베로DB > tablespace에  tablename 존재 유무 검사
    # conn = pyodbc.connect('DSN=TIBERO;UID=UPMS;PWD=UPMS12#$')

    if strCategoryNm == "BDAS":
        # oracle localTest
        conn = get_conn("oracle", "XE", "BDAS", "BDAS12#$", "mapsco.kr:1531/XE", "")

    else:
        conn = pyodbc.connect('DSN=TIBERO;UID=UPMS;PWD=UPMS12#$')
        conn.setencoding(encoding='utf-8')

    strSql = """SELECT MDS_TABLE_SEQ FROM TPBDA020_M WHERE MDS_TABLE_USER='""" + strTableSpaceNm + """' AND MDS_TNAME='""" + strTableNm + """'"""
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
        try:
            rowColumnInfo = getBigDataColumnPK(conn, "", strTableNm)
        except Exception as e:
            print(e)
        print(rowColumnInfo)
    elif strCategoryNm == "THAKSA_NEW":
        # 티베로
        DSNNAME = 'TIBERO'
        DBUSER = 'UPMS'
        DBPWD = 'UPMS12#$'
        conn = pyodbc.connect('DSN=' + DSNNAME + ';UID=' + DBUSER + ';PWD=' + DBPWD)
        # 해당 Table에 대한 컬럼 정보 및 pk정보를 가지고 옴
        rowColumnInfo = getTableColumnPK(conn, DBUSER, strTableNm)
    elif strCategoryNm == "SQL":
        # 동적SQL문
        DSNNAME = 'TIBERO'
        DBUSER = 'UPMS'
        DBPWD = 'UPMS12#$'
        conn = pyodbc.connect('DSN=' + DSNNAME + ';UID=' + DBUSER + ';PWD=' + DBPWD)
        # 해당 Table에 대한 컬럼 정보 및 pk정보를 가지고 옴
        rowColumnInfo = getTableColumnPK(conn, DBUSER, strTableNm)
    elif strCategoryNm == "NoSql":
        # NoSql(MongoDB)에 관한 Collection정보를 가져옴
        DSNNAME = 'TIBERO'
        DBUSER = 'UPMS'
        DBPWD = 'UPMS12#$'
        conn = pyodbc.connect('DSN=' + DSNNAME + ';UID=' + DBUSER + ';PWD=' + DBPWD)
        # 해당 Table에 대한 컬럼 정보 및 pk정보를 가지고 옴
        rowColumnInfo = getNoSqlDocument(conn, strTableSpaceNm, strTableNm)
    elif strCategoryNm == "BDAS":
        # oracle
        DBUSER = 'BDAS'
        conn = get_conn("oracle", "XE", "BDAS", "BDAS12#$", "mapsco.kr:1531/XE", "")
        # 해당 Table에 대한 컬럼 정보 및 pk정보를 가지고 옴
        rowColumnInfo = getTableColumnPK(conn, DBUSER, strTableNm)

    else:
        return HttpResponse("No tspace")

    conn.close()

    # MDS관리 테이블에 정보를 저장

    if strCategoryNm == "BDAS":
        # oracle
        # mdsConn = get_conn("oracle", "XE", "BDAS", "BDAS12#$", "mapsco.kr:1531/XE", "")
        dbuser = "BDAS"
        dbpwd = "BDAS12#$"
        dbhost = "mapsco.kr:1531/XE"

        mdsConn = cx_Oracle.connect(user=dbuser, password=dbpwd, dsn=dbhost)
        cur = mdsConn.cursor()

        cur.execute("""INSERT INTO TPBDA020_M (MDS_TABLE_SEQ, MDS_CATE_NM, MDS_TABLE_USER, MDS_TNAME, MDS_TABLE_DESCRIPTION, MAKE_ID, INPT_ID, INPT_DT, INPT_IP, UPDT_ID, UPDT_DT, UPDT_IP) 
                            VALUES (SEQ_TPBDA020_M.NEXTVAL, :MDS_CATE_NM, :MDS_TABLE_USER, :MDS_TNAME, :MDS_TABLE_DESCRIPTION,:MAKE_ID, :INPT_ID,SYSDATE, :INPT_IP, :UPDT_ID,SYSDATE, :UPDT_IP)""",
                    (strCategoryNm, strTableSpaceNm, strTableNm, strTableComment, strUserId, strUserId, strUserIp,
                     strUserId, strUserIp))

        mdsConn.commit()

    elif strCategoryNm == "UPMS" or strCategoryNm == "SQL" or strCategoryNm == "NoSql":
        # tibero 등 기타
        # MDS관리 테이블에 정보를 저장
        mdsConn = pyodbc.connect('DSN=TIBERO;UID=UPMS;PWD=UPMS12#$')

        # 서버가 원도용가 아닐경우 아래 주석을 풀어주기
        mdsCursor = mdsConn.cursor()
        mdsCursor.execute("""INSERT INTO TPBDA020_M (MDS_TABLE_SEQ, MDS_CATE_NM, MDS_TABLE_USER, MDS_TNAME, MDS_TABLE_DESCRIPTION, MAKE_ID, INPT_ID, INPT_DT, INPT_IP, UPDT_ID, UPDT_DT, UPDT_IP) 
                                VALUES (SEQ_TPBDA020_M.NEXTVAL,?,?,?,?,?,?,SYSDATE,?,?,SYSDATE,?)""",
                          strCategoryNm, strTableSpaceNm, strTableNm, strTableComment, strUserId, strUserId, strUserIp,
                          strUserId, strUserIp)
        mdsConn.commit()

    # MDS관리 컬럼 테이블에 정보를 저장
    if strCategoryNm == "BDAS":
        # oracle
        for it in rowColumnInfo:
            cur.execute("""INSERT INTO TPBDA030_D (MDS_COLUMN_SEQ, MDS_TABLE_SEQ, MDS_COLUMN_NAME, MDS_COLUMN_TYPE, MDS_COLUMN_DESCRIPTION, COLUMN_KEY_YN, COLUMN_ID_SEQ, MAKE_ID, INPT_ID, INPT_DT, INPT_IP, UPDT_ID, UPDT_DT, UPDT_IP)
                                            VALUES (:MDS_COLUMN_SEQ, SEQ_TPBDA020_M.CURRVAL, :MDS_COLUMN_NAME, :MDS_COLUMN_TYPE, :MDS_COLUMN_DESCRIPTION, :COLUMN_KEY_YN, :COLUMN_ID_SEQ, :MAKE_ID, :INPT_ID, SYSDATE, :INPT_IP, :UPDT_ID, SYSDATE, :UPDT_IP)""",
                        (int(it[4]), it[0], it[1], it[2], it[3].strip(), int(it[4]), strUserId, strUserId, strUserIp,
                         strUserId, strUserIp))
            mdsConn.commit()

    elif strCategoryNm == "UPMS" or strCategoryNm == "SQL" or strCategoryNm == "NoSql":
        for it in rowColumnInfo:
            mdsConn.execute("""INSERT INTO TPBDA030_D (MDS_COLUMN_SEQ, MDS_TABLE_SEQ, MDS_COLUMN_NAME, MDS_COLUMN_TYPE, MDS_COLUMN_DESCRIPTION, COLUMN_KEY_YN, COLUMN_ID_SEQ, MAKE_ID, INPT_ID, INPT_DT, INPT_IP, UPDT_ID, UPDT_DT, UPDT_IP)
                                            VALUES (?, SEQ_TPBDA020_M.CURRVAL, ?, ?, ?, ?, ?, ?, ?, SYSDATE, ?, ?, SYSDATE, ?)""",
                            int(it[4]), it[0], it[1], it[2], it[3].strip(), int(it[4]), strUserId, strUserId, strUserIp,
                            strUserId, strUserIp)
            mdsConn.commit()

    elif strCategoryNm == "BIG_DATA":
        for idx, val in enumerate(rowColumnInfo):
            COLUMN_ID_SEQ = idx
            MDS_COLUMN_NAME = val[0]
            COLUMN_KEY_YN = "N"
            MDS_COLUMN_TYPE = val[1]
            MDS_COLUMN_DESCRIPTION = val[2]

            mdsConn.execute("""INSERT INTO TPBDA030_D (MDS_COLUMN_SEQ, MDS_TABLE_SEQ, MDS_COLUMN_NAME, MDS_COLUMN_TYPE, MDS_COLUMN_DESCRIPTION, COLUMN_KEY_YN, COLUMN_ID_SEQ, MAKE_ID, INPT_ID, INPT_DT, INPT_IP, UPDT_ID, UPDT_DT, UPDT_IP)
                                            VALUES (:MDS_COLUMN_SEQ, SEQ_TPBDA020_M.CURRVAL, :MDS_COLUMN_NAME, :MDS_COLUMN_TYPE, :MDS_COLUMN_DESCRIPTION, :COLUMN_KEY_YN, :COLUMN_ID_SEQ, :strUserId, :strUserId, SYSDATE, :strUserIp, :strUserId, SYSDATE, :strUserIp)""",
                            (COLUMN_ID_SEQ, MDS_COLUMN_NAME, MDS_COLUMN_TYPE, MDS_COLUMN_DESCRIPTION, COLUMN_KEY_YN,
                             COLUMN_ID_SEQ, strUserId, strUserId, strUserIp, strUserId, strUserIp))
            mdsConn.commit()

    # Connection 닫기
    mdsConn.close()

    # 결과값을 JSON으로 리턴
    tables = dict()
    tables["status"] = "ok"
    tables["msg"] = "table insert."
    array_dict.append(tables)
    json_array = json.dumps(array_dict)
    return HttpResponse(json_array, content_type="application/json; charset=utf-8")


# 컬럼 정보 및 PK정보를 가지고 옴
def getTableColumnPK(dbConn, tableOwner, tableNm):
    # dbConn.setencoding(encoding='utf-8')
    strSql = """SELECT T2.COLUMN_NAME, T1.DATA_TYPE, T2.COMMENTS
                     , DECODE(NVL(T3.CONSTRAINT_TYPE, 'N'), 'N', 'N', 'Y') AS PK_YN, T1.COLUMN_ID
                FROM ALL_TAB_COLUMNS T1 
                   , ALL_COL_COMMENTS T2 
                   , ( SELECT A.TABLE_NAME, B.COLUMN_NAME, A.CONSTRAINT_NAME, A.CONSTRAINT_TYPE, A.SEARCH_CONDITION
                       FROM USER_CONSTRAINTS A, USER_CONS_COLUMNS B
                       WHERE A.CONSTRAINT_NAME = B.CONSTRAINT_NAME
                         AND A.OWNER='""" + tableOwner + """' AND A.TABLE_NAME ='""" + tableNm + """' AND A.CONSTRAINT_TYPE = 'P'
                     ) T3
                WHERE T1.OWNER = T2.OWNER
                  AND T1.TABLE_NAME = T2.TABLE_NAME
                  AND T1.COLUMN_NAME = T2.COLUMN_NAME
                  AND T1.TABLE_NAME = T3.TABLE_NAME(+)
                  AND T1.COLUMN_NAME = T3.COLUMN_NAME(+)
                  AND T1.OWNER='""" + tableOwner + """' AND T1.TABLE_NAME ='""" + tableNm + """'
                ORDER BY T1.COLUMN_ID"""

    cur = dbConn.execute(strSql)
    retRow = cur.fetchall()

    return retRow


# NoSql Document Key, Key 설명 정보를 가지고 옴
def getNoSqlDocument(dbConn, tableOwner, tableNm):
    print("==" + tableOwner)
    dbConn.setencoding(encoding='utf-8')
    strSql = """SELECT T1.DOCUMENT_KEY AS COLUMN_NAME, 'VARCHAR' AS DATA_TYPE, T1.DOCUMENT_COMMENT AS COMMENTS 
                     , 'N' AS PK_YN, T1.SORT_ORD AS COLUMN_ID
                FROM MCBDA020_D T1 
                   , MCBDA010_M T2 
                WHERE T1.DATABASE_ID = T2.DATABASE_ID
                  AND NVL(T1.USE_YN, 'Y') = 'Y'
                  AND T2.UP_DATABASE_NM='""" + tableOwner + """' AND T2.DATABASE_NM ='""" + tableNm + """'
                ORDER BY T1.SORT_ORD"""
    cur = dbConn.execute(strSql)
    retRow = cur.fetchall()
    print(retRow)

    return retRow


def getBigDataColumnPK(dbConn, tableOwner, tableNm):
    strSql = """DESC """ + tableNm
    cur = dbConn.cursor()
    cur.execute(strSql)
    retRow = cur.fetchall()
    return retRow