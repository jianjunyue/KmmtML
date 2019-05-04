import pandas as pd
import numpy as np
import datetime
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

# https://tianchi.aliyun.com/competition/entrance/231522/information
# https://tianchi.aliyun.com/competition/entrance/1/introduction

path="/Users/jianjun.yue/KmmtML/data/tianchi/天池新人实战赛之[离线赛]/"
# train_item = pd.read_csv(path+"tianchi_fresh_comp_train_item.csv")
# train_user = pd.read_csv(path+"tianchi_fresh_comp_train_user.csv")
train_user=pd.read_csv(path+"tianchi_fresh_comp_train_user_sample.csv")


def time_index(data):
    times=data.replace(" ","").replace("-","")
    return int(times)

train_user["time_index"]=train_user["time"].apply(time_index)
train_user["user_item_index"]=train_user["user_id"]+train_user["item_id"]
print("-----------------")
print(train_user.head())

print("--------------LogisticRegression---------------")
predictors=["user_id","item_id","behavior_type","item_category","time_index","time"]


sql1="select user_item_index,user_id,item_id,behavior_type,item_category,time_index,time from train_user  where behavior_type=1 "
behavior_type_1 = pysqldf(sql1)

sql2="select user_item_index,user_id,item_id,behavior_type,item_category,time_index,time from train_user  where behavior_type=2 "
behavior_type_2 = pysqldf(sql2)

sql3="select user_item_index,user_id,item_id,behavior_type,item_category,time_index,time from train_user  where behavior_type=3 "
behavior_type_3= pysqldf(sql3)

sql4="select user_item_index,user_id,item_id,behavior_type,item_category,time_index,time from train_user  where behavior_type=4 "
behavior_type_4 = pysqldf(sql4)

sql5="select t1.user_item_index,t1.user_id,t1.item_id,t1.time as time1 ,t2.time as time2,t1.behavior_type as type1,t2.behavior_type as type2 from behavior_type_1 t1 left join behavior_type_2 t2 on  t1.user_item_index=t2.user_item_index"
behavior_type_1_4 = pysqldf(sql5)

print(behavior_type_1_4.head())

sql_1_2_3_4=""





