import cx_Oracle
import pyodbc
import os
from pyhive import hive


# dsname is used only in tibero.
# dbhost, dbport is not used in tibero
# hive don't use dbuser, dbpwd
def get_conn(dbtype, dsname, dbuser, dbpwd, dbhost, dbport):
    if dbtype == "hive":
        conn = hive.Connection(host=dbhost, port=dbport, auth="NOSASL", database="default")
        cur = conn.cursor()

        return cur

    elif dbtype == "oracle":
        os.chdir(r'instantclient_19_11')
        os.putenv('NLS_LANG', 'AMERICAN_AMERICA.UTF8')

        db = cx_Oracle.connect(user=dbuser, password=dbpwd, dsn=dbhost)
        cur = db.cursor()

##push test

        ##push test
        ##push test

        return cur

    elif dbtype == "tibero":

        cur = pyodbc.connect('DSN=' + dsname + ';UID=' + dbuser + ';PWD=' + dbpwd)
        # cur.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        # cur.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        # cur.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-8')
        cur.setencoding(encoding='utf-8')

        return cur
    #
    # elif dbtype == "MSSQL":
    #
    #
    # elif dbtype == "MYSQL":
    #
    #
    # elif dbtype == "MARIADB":

