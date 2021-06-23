import json

from django.shortcuts import render
from django.http import HttpResponse
from pyhive import hive
import cx_Oracle
import pyodbc
import os
# Create your views here.


def get_table(request):
    #오라클 연결방법
    # os.chdir(r'C:\instantclient_19_11')
    # os.putenv('NLS_LANG', 'AMERICAN_AMERICA.UTF8')
    #
    #
    # db = cx_Oracle.connect(user="GW_SCFF", password="GWSU12#$", dsn="112.216.254.226:1521/CMELU")
    # cursor = db.cursor()
    #
    # cursor.execute("SELECT * FROM USER_OBJECTS WHERE OBJECT_TYPE='TABLE'")
    # a = cursor.fetchall()
    # cursor.close()
    # db.close()

    # UPMS(티베로), BIG_DATA(하이브)
    # POST로 받은 값이 null이면 GET으로 받음
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    tspace = body['tspace']
    if tspace is None:
        tspace = request.GET.get("tspace")

    if tspace == "BIG_DATA":
        host = "192.168.0.133"
        port = 10000
        conn = hive.Connection(host=host, port=port, auth="NOSASL", database="default")
        cur = conn.cursor()
        cur.execute("show tables")

    elif tspace == "UPMS":
        # 티베로
        DSNNAME = 'TIBERO'
        DBUSER = 'UPMS'
        DBPWD = 'UPMS12#$'

        conn = pyodbc.connect('DSN=' + DSNNAME + ';UID=' + DBUSER + ';PWD=' + DBPWD)
        # conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        # conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        # conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-8')
        conn.setencoding(encoding='utf-8')

        cur = conn.execute("select A.TABLE_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, A.NUM_ROWS NUM_ROWS from ALL_TABLES A, ALL_TAB_COMMENTS B where A.TABLE_NAME = B.TABLE_NAME AND A.OWNER = '"+tspace+"' "+
			"UNION ALL SELECT A.VIEW_NAME TABLE_NAME, A.OWNER TABLESPACE_NAME, B.COMMENTS COMMENTS, 0 NUM_ROWS from ALL_VIEWS A, USER_TAB_COMMENTS B where A.VIEW_NAME = B.TABLE_NAME AND A.OWNER='"+tspace+"' ORDER BY TABLE_NAME")

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
            tables["NUM_ROWS"] = ''

        try:
            tables["NUM_ROWS"] = it[3]
        except IndexError:
            tables["NUM_ROWS"] = ''

        array_dict.append(tables)
        json_array = json.dumps(array_dict)
    return HttpResponse(json_array,  content_type="application/json; charset=utf-8")

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