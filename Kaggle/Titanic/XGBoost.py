# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from sklearn.metrics import make_scorer
from xgboost.sklearn import XGBClassifier

# 网格搜索
# 可以先固定一个参数 最优化后继续调整
# 第一步：确定学习速率和tree_based 给个常见初始值 根据是否类别不平衡调节
# max_depth,min_child_weight,gamma,subsample,scale_pos_weight
# max_depth=3 起始值在4-6之间都是不错的选择。
# min_child_weight比较小的值解决极不平衡的分类问题eg:1
# subsample, colsample_bytree = 0.8: 这个是最常见的初始值了
# scale_pos_weight = 1: 这个值是因为类别十分不平衡。
# 第二步： max_depth 和 min_weight 对最终结果有很大的影响
# 'max_depth':range(3,10,2),
# 'min_child_weight':range(1,6,2)
# 先大范围地粗调参数，然后再小范围地微调。
# 第三步：gamma参数调优
# 'gamma':[i/10.0 for i in range(0,5)]
# 第四步：调整subsample 和 colsample_bytree 参数
# 'subsample':[i/100.0 for i in range(75,90,5)],
# 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
# 第五步：正则化参数调优
# 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
# 'reg_lambda'
# 第六步：降低学习速率
# learning_rate =0.01,


path="train.csv"
data=pd.read_csv(path)
path_test="test.csv"
data_test=pd.read_csv(path_test)

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]


train=data[predictors]
y=data["Survived"]
test=data_test[predictors]


    #多分类准确度评估
def multil_score_proba(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score=0
    for i in range(0,y_true.size) :
        score+=y_pred[i][y_true[i]-1]

    return score/(1.0*y_true.size)

# X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=1)
# # xgb=XGBClassifier(seed=27,learning_rate=0.1,min_child_weight=1,objective= 'multi:softmax')
# xgb = XGBClassifier(seed=27,objective= 'multi:softmax',colsample_bytree=0.4603, gamma=0.0468,
#                              learning_rate=0.05, max_depth=3,
#                              min_child_weight=1.7817, n_estimators=2200,
#                              reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, silent=1,
#                              nthread = -1)
#
# xgb.fit(X_train, y_train)
# preds =xgb.predict_proba(X_test)
# multil_score=multil_score_proba(y_test, preds)
# print(multil_score)


#这里的greater_is_better参数决定了自定义的评价指标是越大越好还是越小越好
score_proba=make_scorer("roc_auc", greater_is_better=True, needs_proba = True)
score_proba="roc_auc"
# print("----------------XGBClassifier-----------------")
# model=XGBClassifier(seed=27,objective= 'multi:softmax',learning_rate=0.1,min_child_weight=1)
# gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1,scoring=score_proba,n_jobs=-1,cv=5)
# gsearch1.fit(train,y)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)
def get_param(key,value):
    if key=='max_depth' or key=='n_estimators' or key=='min_child_weight':
        param = np.array([0])
        param[0]=value
    else:
        param = np.array([0.1])
        param[0] = value
    return param

model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=200,
 max_depth=3,
 min_child_weight=2,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1,
 seed=27)

print("--------------------第二步-----------------------")
#第二步：max_depth([默认6],典型值：3-10) 和 min_child_weight([默认1],典型值：3-10) 参数调优
param_test = {
 'max_depth':range(2,10,1),
 'min_child_weight':range(1,10,1)
}
print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

print("--------------------第三步-----------------------")
#第三步：gamma([默认0]，典型值：0-0.2)参数调优
param_test = {
 'gamma':[i/10.0 for i in range(0,5)]
}
for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)
print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

print("--------------------第四步-----------------------")
#第四步：调整subsample([默认1],典型值：0.5-0.9) 和 colsample_bytree([默认1],典型值：0.5-0.9) 参数
param_test = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)

print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=True,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

print("--------------------第五步-----------------------")
#第五步：正则化参数调优 lambda->reg_lambda([默认1]) , alpha->reg_alpha[默认1]
param_test = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)

print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

print("--------------------第六步-----------------------")
#第六步：降低学习速率 learning_rate =0.01, n_estimators=5000,
param_test = {
 # 'learning_rate':[i/100.0 for i in range(1,20)]
 'learning_rate':[i/200.0 for i in range(16,26)]
}

for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)

print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)


param_test = {
 'n_estimators':range(50,3000,50)
}

for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)

print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)