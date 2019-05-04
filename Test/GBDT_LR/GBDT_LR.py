# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from sklearn.metrics import make_scorer
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py

pre_path="/Users/jianjun.yue/KmmtML/data/kaggle/titanic/"
path="train.csv"
data=pd.read_csv(pre_path+path)
path_test="test.csv"
data_test=pd.read_csv(pre_path+path_test)
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]
train=data[predictors]
y=data["Survived"]
test=data_test[predictors]
submission=pd.DataFrame()
submission["PassengerId"]=data_test["PassengerId"]

# print(test[test.isnull().values==True])


X_train=train
y_train=y
X_test=test

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.5,random_state=1)
X_train,X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

model = GradientBoostingClassifier(
loss='deviance',  ##损失函数默认deviance  deviance具有概率输出的分类的偏差
n_estimators=100, ##默认100 回归树个数 弱学习器个数
learning_rate=0.1,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
max_depth=3,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
subsample=0.8,  ##树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
min_samples_split=2, ##生成子节点所需的最小样本数 如果是浮点数代表是百分比
min_samples_leaf=1, ##叶节点所需的最小样本数  如果是浮点数代表是百分比
max_features=None, ##在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
max_leaf_nodes=None, ##叶节点的数量 None不限数量
min_impurity_decrease=1e-7, ##停止分裂叶子节点的阈值
verbose=0,  ##打印输出 大于1打印每棵树的进度和性能
warm_start=False, ##True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
random_state=0  ##随机种子-方便重现
)##多类别回归建议使用随机森林

model.fit(X_train,y_train)
y_pred=model.predict(X_test)


grd_enc = OneHotEncoder()
grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
grd_enc.fit(model.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(model.apply(X_test)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(model.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)



# submission["Survived"]=y_pred
# submission.to_csv(pre_path+'GBDT_LR/GradientBoostingClassifier.csv',index=None)
# accuracy_score=metrics.accuracy_score(y_pred,y_test)
# print("GradientBoostingClassifier : ",accuracy_score)

print(submission.head(100))