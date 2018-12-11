import pandas as pd
from Utils.FileUtil import FileUtil
from Utils.Util import Util
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# 机器学习特征工程
# https://zhuanlan.zhihu.com/p/26444240

path=FileUtil.dataKagglePath("titanic/")+"train.csv"
data=pd.read_csv(path)
path_test=FileUtil.dataKagglePath("titanic/")+"test.csv"
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

onehot_title=onehot_encoding(train,"Title")
train = train.join(onehot_title)
print(train.head())

# 3.1.3. 聚类编码
# 先把一个特征值进行K-means聚成类别编码，再对聚类编码进行独热编码？

# 3.1.4. 平均数编码（mean encoding）