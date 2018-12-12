#!/usr/bin/env python
#coding:utf8
import pandas as pd
from Algorithm.MeanEncoder import MeanEncoder
from Utils.FileUtil import FileUtil
from Algorithm.PCA import PCA

categorical_features=""
# aa=MeanEncoder(categorical_features)
# path= FileUtil.dataKagglePath()
# print(path)


path=FileUtil.dataKagglePath("titanic/")+"train.csv"
data=pd.read_csv(path)
path_test=FileUtil.dataKagglePath("titanic/")+"test.csv"
data_test=pd.read_csv(path_test)
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]
train=data[predictors]
y=data["Survived"]
test=data_test[predictors]

MeanEnocodeFeature = ["Title","Age"] #声明需要平均数编码的特征
ME = MeanEncoder(MeanEnocodeFeature) #声明平均数编码的类
trans_train = ME.fit_transform(train,y)#对训练数据集的X和y进行拟合
test_trans = ME.transform(test)#对测试集进行编码

# print("----------------trans_train-----------------")
# print(trans_train)
# print("----------------test_trans-----------------")
# print(test_trans)

print(len(train.columns))
pca = PCA(train.fillna(0))
X_reduce_feature = pca.reduce_dimension()
print("----------------X_reduce_feature-----------------")
print(X_reduce_feature)