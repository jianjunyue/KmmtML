import pandas as pd
import numpy as np
import datetime
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

# https://tianchi.aliyun.com/competition/entrance/231522/information
# https://tianchi.aliyun.com/competition/entrance/1/introduction

path="/Users/jianjun.yue/KmmtML/data/tianchi/天池新人实战赛之[离线赛]/"
train_item = pd.read_csv(path+"tianchi_fresh_comp_train_item.csv")
# train_user = pd.read_csv(path+"tianchi_fresh_comp_train_user.csv")
train_user_sample=pd.read_csv(path+"tianchi_fresh_comp_train_user_sample.csv")
print(train_item.head())
print(train_user_sample.head())
# train_user_sample=train_user.sample(n=20000)
# train_user_sample.to_csv(path+"tianchi_fresh_comp_train_user_sample.csv")
def weekday(data):
    times=data.split(" ")
    hour= int(times[1])
    day=times[0]
    startTime = datetime.datetime.strptime(day, '%Y-%m-%d').date()
    return startTime.weekday()
def hour(data):
    times=data.split(" ")
    hour= int(times[1])
    day=times[0]
    startTime = datetime.datetime.strptime(day, '%Y-%m-%d').date()
    return hour

train_user_sample["hour"]=train_user_sample["time"].apply(hour)
train_user_sample["week"]=train_user_sample["time"].apply(weekday)
print("-----------------")
print(train_user_sample.head())

print("--------------LogisticRegression---------------")
predictors=["user_id","item_id","behavior_type","item_category","hour","week"]



sql="select user_id,item_id from train_user_sample where behavior_type=3 and  time like '2014-12-18%' "
table = pysqldf(sql)
print(table.head())
table.to_csv(path+"tianchi_fresh_comp_train_user_sample_submission.csv",index=False)









