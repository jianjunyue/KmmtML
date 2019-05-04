import pandas as pd
import numpy as np
import datetime
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

df=pd.DataFrame()
value1=["1","2","2","2","3","3","3","2","2","2","2","4","4","4","4","4","4","5","6","6","5"]
value2=["1","1","2","1","1","2","3","1","2","3","4","2","2","3","2","3","4","3","3","4","4"]
value3=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]
df["user_id"]=value1
df["type"]=value2
df["time"]=value3

dict_group={}
dict_type={}
print(type(dict_group))
list=[]
for arr in df.values:
    arr_temp=["0","0","0","0"]
    type=dict_type.get(arr[0])
    group=dict_group.get(arr[0])
    if type==None:
        dict_type[arr[0]]=arr[1]
        dict_group[arr[0]]=0
        group=0
    elif int(arr[1])<=int(type):
        group=group+1
        dict_type[arr[0]] = arr[1]
        dict_group[arr[0]] = group

    arr_temp[0]=arr[0]
    arr_temp[1]=arr[1]
    arr_temp[2]=arr[2]
    arr_temp[3]=str(group)
    list.append(arr_temp)
df_temp=pd.DataFrame(list)

df["group"]=df_temp[3]
df["user_id_group"]=df["user_id"]+"_"+df["group"]
df["time"]=df["time"].astype(int)

print(df.head(100))

sql1="select user_id_group,user_id,type,time from df  where type=1 "
behavior_type_1 = pysqldf(sql1)
sql2="select user_id_group,user_id,type,time from df  where type=2 "
behavior_type_2 = pysqldf(sql2)
sql3="select user_id_group,user_id,type,time from df  where type=3 "
behavior_type_3 = pysqldf(sql3)
sql4="select user_id_group,user_id,type,time from df  where type=4 "
behavior_type_4 = pysqldf(sql4)
# print(behavior_type_1.head(5))
# print(behavior_type_2.head(5))
# print(behavior_type_3.head(5))
# print(behavior_type_4.head(5))

sql_1_2_3_4="select t1.user_id_group, t1.user_id,t1.type as type_1,t2.type as type_2,t3.type as type_3,t4.type as type_4,t1.time as time_1,t2.time as time_2,t3.time as time_3,t4.time as time_4 from  behavior_type_1 t1 inner join behavior_type_2 t2  inner join behavior_type_3 t3  inner join behavior_type_4 t4 where t1.user_id=t2.user_id and t1.user_id=t3.user_id and t1.user_id=t4.user_id and  t2.time>t1.time and t3.time>t2.time and t4.time>t3.time and t1.user_id_group=t2.user_id_group and t1.user_id_group=t3.user_id_group and t1.user_id_group=t4.user_id_group"
behavior_type_1_2_3_4 = pysqldf(sql_1_2_3_4)
print(behavior_type_1_2_3_4.head(5))

sql_1_2_3="select t1.user_id_group, t1.user_id,t1.type as type_1,t2.type as type_2,t3.type as type_3,t1.time as time_1,t2.time as time_2,t3.time as time_3  from  behavior_type_1 t1 inner join behavior_type_2 t2  inner join behavior_type_3 t3  where t1.user_id=t2.user_id and t1.user_id=t3.user_id  and  t2.time>t1.time and t3.time>t2.time and t1.user_id_group=t2.user_id_group and t1.user_id_group=t3.user_id_group and t1.user_id_group not in (select user_id_group from behavior_type_1_2_3_4) "
behavior_type_1_2_3 = pysqldf(sql_1_2_3)
print(behavior_type_1_2_3.head(5))

sql_1_2="select t1.user_id_group, t1.user_id,t1.type as type_1,t2.type as type_2,t1.time as time_1,t2.time as time_2 from  behavior_type_1 t1 inner join behavior_type_2 t2 where t1.user_id=t2.user_id  and  t2.time>t1.time  and t1.user_id_group=t2.user_id_group  and (t1.user_id_group not in (select user_id_group from behavior_type_1_2_3_4)) and (t1.user_id_group not in (select user_id_group from behavior_type_1_2_3)) "
behavior_type_1_2 = pysqldf(sql_1_2)
print(behavior_type_1_2.head(5))

sql_1="select t1.user_id_group, t1.user_id,t1.type as type_1,t1.time as time_1 from  behavior_type_1 t1  where  (t1.user_id_group not in (select user_id_group from behavior_type_1_2_3_4)) and (t1.user_id_group not in (select user_id_group from behavior_type_1_2_3)) and (t1.user_id_group not in (select user_id_group from behavior_type_1_2)) "
behavior_type_1 = pysqldf(sql_1)
print(behavior_type_1.head(5))

sql_2_3_4="select t2.user_id_group, t2.user_id,t2.type as type_2,t3.type as type_3,t4.type as type_4,t2.time as time_2,t3.time as time_3,t4.time as time_4 from  behavior_type_2 t2  inner join behavior_type_3 t3  inner join behavior_type_4 t4 where t2.user_id=t3.user_id and t2.user_id=t4.user_id and t3.time>t2.time and t4.time>t3.time and t2.user_id_group=t3.user_id_group and t2.user_id_group=t4.user_id_group and  t2.user_id_group not in (select user_id_group from behavior_type_1_2_3_4)"
behavior_type_2_3_4 = pysqldf(sql_2_3_4)
print(behavior_type_2_3_4.head(5))