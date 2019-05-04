import pandas as pd
import numpy as np
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

path="/Users/jianjun.yue/KmmtML/data/比赛/腾讯广告算法大赛2019/"

# 历史曝光日志数据文件 (totalExposureLog.out)
totalExposureLog = pd.read_csv(path+"totalExposureLog_sample.out", sep='\t',header=None,names = ["id", "log_time", "ad_place_id", "user_id","ad_id","ad_size_enum","ad_price_bid","ad_pctr","ad_quality_ecpm","ad_totalEcpm"])
print("------------------totalExposureLog-------------------")
print(totalExposureLog.head())

# 广告操作数据 (ad_operation.dat)
ad_operation = pd.read_csv(path+"ad_operation.dat", sep='\t',header=None,names = ["ad_id","ad_operation_time","ad_operation_type","ad_edit_type","ad_value"])
print("------------------ad_operation-------------------")
print(ad_operation.head())

# 广告静态数据 (ad_static_feature.out)
ad_static_feature = pd.read_csv(path+"ad_static_feature.out", sep='\t',header=None,names = ["ad_id","ad_create_time","ad_account_id","ad_promote_goods_id","ad_promote_goods_type","ad_industry_id","ad_size"])
print("------------------ad_static_feature-------------------")
print(ad_static_feature.head())

# 用户特征属性文件（user_data）
# user_data = pd.read_csv(path+"user_data", sep='\t',header=None,low_memory=False,names = ["user_id","age","gender","area","status","education","consuptionAbility","device","work","connectionType","behavior"])
# print("------------------user_data-------------------")
# print(user_data.head())

#
# sql="select id,log_time,ad_place_id,totalExposureLog.user_id,totalExposureLog.ad_id,ad_size_enum,ad_price_bid,ad_pctr,ad_quality_ecpm,ad_totalEcpm,age,gender,area,status,education,education,consuptionAbility,device,work,connectionType,behavior,ad_operation_time,ad_operation_type,ad_edit_type,ad_value,ad_create_time,ad_account_id,ad_promote_goods_id,ad_promote_goods_type,ad_industry_id,ad_size  from totalExposureLog left join ad_static_feature on totalExposureLog.ad_id=ad_static_feature.ad_id left join ad_operation on totalExposureLog.ad_id=ad_operation.ad_id left join user_data on totalExposureLog.user_id=user_data.user_id"

sql="select id,log_time,ad_place_id,totalExposureLog.user_id,totalExposureLog.ad_id,ad_size_enum,ad_price_bid,ad_pctr,ad_quality_ecpm,ad_totalEcpm,ad_operation_time,ad_operation_type,ad_edit_type,ad_value,ad_create_time,ad_account_id,ad_promote_goods_id,ad_promote_goods_type,ad_industry_id,ad_size  from totalExposureLog left join ad_static_feature on totalExposureLog.ad_id=ad_static_feature.ad_id left join ad_operation on totalExposureLog.ad_id=ad_operation.ad_id"
print("------------------table_temp-------------------")
table_temp = pysqldf(sql)
print(table_temp.head())


table_temp.to_csv(path+"temp/log_left_join_ad_info.csv",index=False)
