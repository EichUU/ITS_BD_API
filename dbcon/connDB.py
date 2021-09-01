import cx_Oracle
import pyodbc
import pymysql
import mariadb
import pymssql
import os
from pyhive import hive


# dsname is used only in tibero, mariadb.
# dbhost, dbport is not used in tibero
# hive don't use dbuser, dbpwd

def get_conn(dbtype, dsname, dbuser, dbpwd, dbhost, dbport):
    cur = None
    if dbtype == "hive":
        conn = hive.Connection(host=dbhost, port=dbport, auth="NOSASL", database="default")
        cur = conn.cursor()

    elif dbtype == "oracle":
        os.chdir(r'instantclient_19_11')
        os.putenv('NLS_LANG', 'AMERICAN_AMERICA.UTF8')

        db = cx_Oracle.connect(user=dbuser, password=dbpwd, dsn=dbhost)
        cur = db.cursor()

    elif dbtype == "tibero":

        cur = pyodbc.connect('DSN=' + dsname + ';UID=' + dbuser + ';PWD=' + dbpwd)
        # cur.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        # cur.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        # cur.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-8')
        if os.name != "nt":
            cur.setencoding(encoding='utf-8')

    elif dbtype == "mssql":
        conn = pymssql.connect(host=dbhost, database=dsname, port=dbport, charset='utf8')
        cur = conn.cursor()

    elif dbtype == "mysql":
        conn = pymysql.connect(host=dbhost, user=dbuser, password=dbpwd, port=int(dbport), db=dsname, charset='utf8')
        cur = conn.cursor()

    elif dbtype == "mariadb":
        conn = mariadb.connect(host=dbhost, user=dbuser, password=dbpwd, port=int(dbport), database=dsname)
        cur = conn.cursor()

    return cur

