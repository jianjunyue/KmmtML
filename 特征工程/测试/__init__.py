#!/usr/bin/env python
#coding:utf8
import pandas as pd
from KmmtML.Algorithm.MeanEncoder import MeanEncoder
from KmmtML.Utils.FileUtil import FileUtil
from KmmtML.Algorithm.PCA import PCA

from sklearn2pmml import sklearn2pmml

categorical_features=""
# aa=MeanEncoder(categorical_features)
# path= FileUtil.dataKagglePath()
# print(path)

titanicPath=FileUtil.dataKagglePath("titanic/")
path=FileUtil.dataKagglePath("titanic/")+"train.csv"
data=pd.read_csv(path)
path_test=FileUtil.dataKagglePath("titanic/")+"test.csv"
data_test=pd.read_csv(path_test)
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]
train=data[predictors]
y=data["Survived"]
test=data_test[predictors]

# MeanEnocodeFeature = ["Title","Age"] #声明需要平均数编码的特征
# ME = MeanEncoder(MeanEnocodeFeature) #声明平均数编码的类
# trans_train = ME.fit_transform(train,y)#对训练数据集的X和y进行拟合
# test_trans = ME.transform(test)#对测试集进行编码


# 3.1.4. 平均数编码（mean encoding）
MeanEnocodeFeature = ["Title","Age"] #声明需要平均数编码的特征
ME = MeanEncoder(MeanEnocodeFeature) #声明平均数编码的类
train_mean_encoding = ME.fit_transform(train,y)#对训练数据集的X和y进行拟合
test_mean_encoding = ME.transform(test)#对测试集进行编码

train_mean_encoding["Survived"]=data["Survived"]
test_mean_encoding["PassengerId"]=data_test["PassengerId"]
train_mean_encoding.to_csv(titanicPath+'fe/train_mean_encoding.csv',index=None)
test_mean_encoding.to_csv(titanicPath+'fe/test_mean_encoding.csv',index=None)

print("----------------trans_train-----------------")
# print(trans_train)
# print("----------------test_trans-----------------")
# print(test_trans)

# print(len(train.columns))
# pca = PCA(train.fillna(0))
# X_reduce_feature = pca.reduce_dimension()
# print("----------------X_reduce_feature-----------------")
# print(X_reduce_feature)

print("-----------特征哈希编码方案------------")
# import pandas as pd
# from category_encoders.hashing import HashingEncoder
#
# df_X = pd.DataFrame([1,2,3,4,1,2,4,5,8,7,66,2,24,5,4,1,2,111,1,31,3,23,13,24],columns=list("A"))
#
# he = HashingEncoder(cols=["A"],return_df=True)
# df_X = he.fit_transform(df_X)