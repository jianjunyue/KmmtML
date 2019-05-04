import pandas as pd
import numpy as np
from pandasql import sqldf
import time

pysqldf = lambda q: sqldf(q, globals())

path="/Users/jianjun.yue/KmmtML/data/比赛/腾讯广告算法大赛2019/"

# 历史曝光日志数据文件 (totalExposureLog.out)
filename="totalExposureLog.out" #totalExposureLog_sample.out
totalExposureLog = pd.read_csv(path+filename, sep='\t',header=None,names = ["id", "log_time", "ad_place_id", "user_id","ad_id","ad_size_enum","ad_price_bid","ad_pctr","ad_quality_ecpm","ad_totalEcpm"])
print("------------------totalExposureLog-------------------")
print(totalExposureLog.head())

def get_time(timestamp):
    h = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp))
    return h

def get_day(timestamp):
    h = time.strftime('%Y%m%d', time.gmtime(timestamp))
    return int(h)



# totalExposureLog["log_base_time"]=totalExposureLog["log_time"].apply(get_time)
totalExposureLog["log_day"]=totalExposureLog["log_time"].apply(get_day)

print("------------------totalExposureLog-------------------")
print(totalExposureLog.head())

#
# sql="select max(log_day) from totalExposureLog "
# print("------------------table_temp-------------------")
# table_temp = pysqldf(sql)
# print(table_temp.head())

sql="select ad_id from totalExposureLog where log_day=20190217 "
#
print("------------------table_20190217-------------------")
table_20190217 = pysqldf(sql)
print(table_20190217.head())

table_20190217.to_csv(path+"temp/totalExposureLog_20190217.csv",index=False)

#
# sql="select ad_id,count(ad_id) as ad_id_count from table_20190217  group by ad_id "
#
# print("------------------table_20190217_group-------------------")
# table_20190217_group = pysqldf(sql)
# print(table_20190217_group.head())
#
# table_20190217_group.to_csv(path+"temp/log_ad_id_count.csv",index=False)
