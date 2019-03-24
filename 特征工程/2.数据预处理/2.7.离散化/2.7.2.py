from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from KmmtML.BuildFeature.BuildFeatureBase import BuildFeatureBase
from KmmtML.Data.FileData import FileData


# 2.7 离散化
# 2.7.2 连续数据离散-等宽离散，等频离散，聚类离散。  把数据按不同区间划分（等宽划分或等频划分），聚类编码/按层次进行编码
train=FileData.getData("titanic/","train")
df=pd.DataFrame()
value1=["1","2","3"]
value2=["mall","room","group"]
df["id"]=value1
df["gender"]=value2
print(df[["id","gender"]])
print(type(df[["id","gender"]]))
print("------")
# s= BuildFeatureBase.bin_point(df["gender"])
# print(s)
# train=FileData.getData("titanic/","train")
#
# s= BuildFeatureBase.bin_point(train["Age"],10)
# print(s)
s= BuildFeatureBase.fe_quantile(train["Age"])
print(s)
print(type(s))


print("--  bin_quantile ----")
quantile=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
s= BuildFeatureBase.bin_quantile(train["Age"],quantile)
print(s)
print(type(s))