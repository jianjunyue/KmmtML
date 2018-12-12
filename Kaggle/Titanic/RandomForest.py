# -*- coding: utf-8 -*-
import pandas as pd # coding=UTF-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

path="train.csv"
data=pd.read_csv(path)
path_test="test.csv"
data_test=pd.read_csv(path_test)

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]


train=data[predictors]
y=data["Survived"]
test=data_test[predictors]

score_proba="roc_auc"

def get_param(key,value):
    if key=='max_depth' or key=='min_samples_split' or key=='min_samples_leaf' or key=='max_features':
        param = np.array([0])
        param[0]=value
    else:
        param = np.array([0.1])
        param[0] = value
    return param

print("--------------------第一步-----------------------")
model = RandomForestClassifier(
    n_estimators=60,
    max_depth=7,
    min_samples_split=50,
    min_samples_leaf=10,
    max_features=9,
    oob_score=True,
    random_state=10,
    n_jobs=-1)

print("--------------------第二步-----------------------")
#第二步：max_depth([默认6],典型值：3-10) 和 min_child_weight([默认1],典型值：3-10) 参数调优
param_test = {
 'max_depth':range(2,10,1),
 'min_samples_split':range(50,201,20)
}
print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

print("--------------------第三步-----------------------")
#第三步：对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
param_test = {
 'min_samples_split':range(80,150,20),
 'min_samples_leaf':range(10,60,10)
}
for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)
print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

print("--------------------第四步-----------------------")
#第四步：再对最大特征数max_features做调参
param_test = {
 'max_features':range(3,11,2)
}

for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)

print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)


param_test = {
 # 'n_estimators':range(50,3000,50)
    'n_estimators': range(10, 100, 10)
}

for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)

print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)