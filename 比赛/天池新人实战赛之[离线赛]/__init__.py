import pandas as pd
import numpy as np
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())


path="/Users/jianjun.yue/KmmtML/data/tianchi/天池新人实战赛之[离线赛]/"
train_user = pd.read_csv(path+"tianchi_fresh_comp_train_user.csv")
sql="select user_id,item_id,behavior_type,user_geohash,item_category,time from train_user where behavior_type=3  and  time like '2014-12-18%'   "
table = pysqldf(sql)
print(table.head())
table.to_csv(path+"tianchi_fresh_comp_train_user_submission.csv",index=False)


