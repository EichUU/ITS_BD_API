import json
import datetime
import happybase
import pandas as pd
import pyhive
from django.http import HttpResponse

from dbcon.connDB import get_conn

hbase_host = '192.168.0.193'
hbase_port = '9090'
hive_host = '192.168.0.193'
hive_port = '10000'

#cx_Oracle.connect()
# tableName = 테이블 명
# cf : 컬럼명과 컬럼패밀리 1 : 1 매칭
#      query 컬럼명과 컬럼패밀리 순서가 맞아야함
#     ex){"YY" : "cf1", "STD_NO" : "cf1", "SHYR" : "cf2"}
# query : 조회 쿼리 (ROWKEY 컬럼명이 무조건 들어가야함 => HBASE 테이블에서 키 값이 되는 컬럼)
def hbaseSchedule(request):
    body_unicode = request.body.decode('utf-8')
    if len(body_unicode) != 0:
        body = json.loads(body_unicode)
        tableName = body['tableName']
        cf = body['cf']
        query = body['query']

    else:
        tableName = request.GET.get("tableName")
        cf = request.GET.get("cf")
        query = request.GET.get("query")

    conn = get_conn("oracle", "THAKSA_NEW", "THAKSA_NEW", "Thaksa32!4new", "192.168.0.13:1521/CMELU", "1521")

    cur = conn.execute(query)

    result = pd.DataFrame(cur)
    columns = []
    columnFamily = []

    for i in cf:
        columns.append(i)
        columnFamily.append(cf[i])

    result.columns = columns

    res = hbaseInsert(tableName, result, columnFamily)

    if res.get("SUCCESS_YN") == "Y":
        resp = HttpResponse({'SUCCESS_YN': 'Y'})
        resp.headers['Content-Type'] = "text/plain; charset=utf-8"
        resp.status_code = 200
        return resp
    else:
        resp = HttpResponse("exception occurs.")
        resp.headers['Content-Type'] = "text/plain; charset=utf-8"
        resp.status_code = 400
        return resp


# tableName = 테이블 명
# df = 데이터프레임(저장 데이터, 첫번째에 ROWKEY라는 열을 넣어줘야 함.(PK))
# cf = 컬럼패밀리(df header랑 1:1 매칭)
def hbaseInsert(tableName, df, cfDf):
    # hbase 연결
    happybase_connection = happybase.Connection(hbase_host)
    table = happybase_connection.table(tableName)
    df = df.astype('string')

    for i in range(df.shape[0]):
        tmpStr = ""
        for j in range(len(df.columns)):
            if j > 0:
                if j == 1:
                    tmpStr += "{'" + cfDf[j] + ":" + str(df.columns[j]) + "': '" + str(df[df.columns[j]][i]) + "'"
                if j != 0:
                    tmpStr += ", '" + cfDf[j] + ":" + str(df.columns[j]) + "': '" + str(df[df.columns[j]][i]) + "'"
                if j == len(df.columns) - 1:
                    tmpStr += ", '" + cfDf[j] + ":" + str(df.columns[j]) + "': '" + str(df[df.columns[j]][i]) + "'}"
        try:
            table.put(df['ROWKEY'][i], eval(tmpStr))

        except Exception as e:
            break
            return {'SUCCESS_YN': 'N'}

    return {'SUCCESS_YN': 'Y'}


#df = ROWKEY 만들고자 하는 데이터프레임
#arrPk = 고유값이 될 수 있는 컬럼이름의 배열
def makeRowKey(df, arrPk):
    df['ROWKEY'] = ''
    for i in range(len(arrPk)):
        df['ROWKEY'] += df[arrPk[i]]

    return df


# conn = cx_Oracle.connect("THAKSA_NEW", "Thaksa32!4new", "192.168.102.61:1527/ORA11g")
# conn = conn.cursor()
#
# cf = {"ROWKEY":"cf1", "YY":"cf1", "SHTM":"cf1", "SCHL_KIND_GB":"cf1", "SCHL_KIND_NM":"cf1", "MJ_CD":"cf1", "MJ_NM":"cf1", "PER_SCHL_SUM_AMT":"cf1", "YYSHTM":"cf1"}
#
# query = """SELECT *  FROM VW_SCHL_KIND_MJ_PER"""
#
# #query = """SELECT * FROM VW_SCHL_ANALYZE_KIND_SUM"""
# #query = """SELECT * FROM VW_STD_GRAD_ANALYZE"""
# #query = """SELECT * FROM VW_GRAD_ANALYZE"""
# #
# cur = conn.execute(query)
# print(cur)
# result = pd.DataFrame(cur)
# tableName = "VW_SCHL_KIND_MJ_PER"
#
# columns = []
# columnFamily = []
# #
# for i in cf:
#    columns.append(i)
#    columnFamily.append(cf[i])
# #
# result.columns = columns
# ##
# res = hbaseInsert(tableName, result, columnFamily)
# ##
# #if res.get("SUCCESS_YN") == "Y":
# #    resp = HttpResponse({'SUCCESS_YN': 'Y'})
# ##    resp.headers['Content-Type'] = "text/plain; charset=utf-8"
# ##    resp.status_code = 200
# ##    return resp
# ##else:
# ##    resp = HttpResponse("exception occurs.")
# ##    resp.headers['Content-Type'] = "text/plain; charset=utf-8"
# ##    resp.status_code = 400
# ##    return resp
# print(res)
#
# listFname = ['VW_SCHL_ANALYZE_MASTER.xlsx', 'VW_STD_GRAD_ANALYZE.xlsx', 'VW_STD_GRAD_YY_ANALYZE.xlsx', 'VW_STD_SCHL_SUM.xlsx', 'VW_STD_SHREG_ANALYZE.xlsx']
# #listFname = ['VW_STD_GRAD_YY_ANALYZE.xlsx']
#
# for i in listFname:
#     df = pd.read_excel(r'C:/Users/sunho/Desktop/' + i)
#     cf = {}
#     for j in df.columns:
#         cf.update({j : "cf1"})
#         print(cf)
#
#     tableName = i.split(".")[0]
#
#     columns = []
#     columnFamily = []
#
#     for j in cf:
#        columns.append(j)
#        columnFamily.append(cf[j])
#
#     res = hbaseInsert(tableName, df, columnFamily)

#
