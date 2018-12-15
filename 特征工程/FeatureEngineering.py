import pandas as pd
from Utils.FileUtil import FileUtil
from Utils.Util import Util
from Algorithm.MeanEncoder import MeanEncoder
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# 机器学习特征工程
# https://zhuanlan.zhihu.com/p/26444240
titanicPath=FileUtil.dataKagglePath("titanic/")
path=titanicPath+"train.csv"
data=pd.read_csv(path)
path_test=titanicPath+"test.csv"
data_test=pd.read_csv(path_test)
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]
train=data[predictors]
y=data["Survived"]
test=data_test[predictors]

# print(train.head())
#    PassengerId  Survived  Pclass  Sex  Age  SibSp  Parch     Fare  Embarked  \
# 0            1         0       3    0   22      1      0   7.2500         0
# 1            2         1       1    1   38      1      0  71.2833         1
#
#    FamilySize  NameLength  Title
# 0           1          23      1
# 1           1          51      3


# 3. 特征工程：处理已有特征
# 3.1. 类别特征
# 3.1.1. 自然数编码
# 3.1.2. 独热编码（One-hot Encoding）
def onehot_encoding(df,column_name):
    lb = preprocessing.LabelBinarizer()
    # gender 改为 0-1 数值
    # df["gender"]= lb.fit_transform(df['gender'])
    column_name_dummies  = pd.get_dummies(df[column_name])
    column_name_dummies.rename(columns=Util.rename_columns(column_name,"_",column_name_dummies.columns), inplace = True)
    return column_name_dummies

onehot_title_data=onehot_encoding(data,"Title")
onehot_title_test=onehot_encoding(data_test,"Title")
onehot_encoding_columns = list((set(onehot_title_data.columns).union(set(onehot_title_test.columns)))^(set(onehot_title_data.columns)^set(onehot_title_test.columns)))

train_onehot_encoding = data.join(onehot_title_data[onehot_encoding_columns])
test_onehot_encoding = data_test.join(onehot_title_test[onehot_encoding_columns])
# print(train_onehot_encoding.head())
train_onehot_encoding.to_csv(titanicPath+'fe/train_onehot_encoding.csv',index=None)
test_onehot_encoding.to_csv(titanicPath+'fe/test_onehot_encoding.csv',index=None)

# 3.1.3. 聚类编码
# 先把一个特征值进行K-means聚成类别编码，再对聚类编码进行独热编码？

# 3.1.4. 平均数编码（mean encoding）
def meanEnocode(train, y,meanEnocodeFeature):
    # meanEnocodeFeature = ["Title", "Age"]  # 声明需要平均数编码的特征
    ME = MeanEncoder(meanEnocodeFeature)  # 声明平均数编码的类
    train_mean_encoding = ME.fit_transform(train, y)  # 对训练数据集的X和y进行拟合
    test_mean_encoding = ME.transform(test)  # 对测试集进行编码
    return train_mean_encoding,test_mean_encoding

train_mean_encoding,test_mean_encoding=meanEnocode(train, y,["Title", "Age"])

train_mean_encoding["Survived"]=data["Survived"]
train_mean_encoding["PassengerId"]=data["PassengerId"]
test_mean_encoding["PassengerId"]=data_test["PassengerId"]
train_mean_encoding.to_csv(titanicPath+'fe/train_mean_encoding.csv',index=None)
test_mean_encoding.to_csv(titanicPath+'fe/test_mean_encoding.csv',index=None)


