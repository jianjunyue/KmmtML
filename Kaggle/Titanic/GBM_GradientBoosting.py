import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV

#https://blog.csdn.net/han_xiaoyang/article/details/52663170
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
#https://github.com/lytforgood/MachineLearningTrick

from sklearn.metrics import make_scorer


path="train.csv"
data=pd.read_csv(path)
path_test="test.csv"
data_test=pd.read_csv(path_test)

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]


train=data[predictors]
y=data["Survived"]
test=data_test[predictors]

#这里的greater_is_better参数决定了自定义的评价指标是越大越好还是越小越好
score_proba="roc_auc"

def get_param(key,value):
    if(type(value)==str):
        param = []
        param.append(value)
        print(param)
        return param
    if key=='max_depth' or key=='n_estimators' or key=='min_samples_split' or key=='min_samples_leaf' or key=='max_features':
        param = np.array([0])
        param[0]=value
    else:
        param = np.array([0.1])
        param[0] = value
    return param

model = GradientBoostingClassifier(
loss='deviance',  ##损失函数默认deviance  deviance具有概率输出的分类的偏差
n_estimators=100, ##默认100 回归树个数 弱学习器个数
learning_rate=0.1,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
max_depth=3,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
subsample=1,  ##树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
min_samples_split=2, ##生成子节点所需的最小样本数 如果是浮点数代表是百分比
min_samples_leaf=1, ##叶节点所需的最小样本数  如果是浮点数代表是百分比
max_features=None, ##在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
max_leaf_nodes=None, ##叶节点的数量 None不限数量
min_impurity_decrease=1e-7, ##停止分裂叶子节点的阈值
verbose=0,  ##打印输出 大于1打印每棵树的进度和性能
warm_start=False, ##True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
random_state=0  ##随机种子-方便重现
)##多类别回归建议使用随机森林

# est = GradientBoostingRegressor(
# loss='ls',      ##默认ls损失函数'ls'是指最小二乘回归lad'（最小绝对偏差）'huber'是两者的组合
# n_estimators=100, ##默认100 回归树个数 弱学习器个数
# learning_rate=0.1,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
# max_depth=3,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
# subsample=1,  ##用于拟合个别基础学习器的样本分数 选择子样本<1.0导致方差的减少和偏差的增加
# min_samples_split=2, ##生成子节点所需的最小样本数 如果是浮点数代表是百分比
# min_samples_leaf=1, ##叶节点所需的最小样本数  如果是浮点数代表是百分比
# max_features=None, ##在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
# max_leaf_nodes=None, ##叶节点的数量 None不限数量
# min_impurity_split=1e-7, ##停止分裂叶子节点的阈值
# verbose=0,  ##打印输出 大于1打印每棵树的进度和性能
# warm_start=False, ##True在前面基础上增量训练 False默认擦除重新训练 增加树
# random_state=0  ##随机种子-方便重现
# )

##网格搜索调整超参数
# clf=GridSearchCV(
# estimator, ##模型
# param_grid, ##参数字典或者字典列表
# scoring=None,  ##评价分数的方法
# fit_params=None, ##fit的参数 字典
# n_jobs=1, ##并行数  -1全部启动
# iid=True,  ##每个cv集上等价
# refit=True,  ##使用整个数据集重新编制最佳估计量
# cv=None,   ##几折交叉验证None默认3
# verbose=0, ##控制详细程度：越高，消息越多
# pre_dispatch='2*n_jobs',  ##总作业的确切数量
# error_score='raise',  ##错误时选择的分数
# return_train_score=True   ##如果'False'，该cv_results_属性将不包括训练得分
# )

# param_test1 = {'n_estimators':range(20,81,10)}
# param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
# param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}

print("--------------------第二步-----------------------")
#第二步：调节树参数. 结果影响最大的参数应该优先调节
# 树参数可以按照这些步骤调节：
# 调节max_depth和 num_samples_split(min_samples_split)
# 调节min_samples_leaf
# 调节max_features
param_test = {
 'max_depth':range(1,10,2),
 # 'min_samples_split':range(200,1001,200)
}
print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

param_test = {
 # 'min_samples_split':range(1000,2100,200),
 # 'min_samples_leaf':range(30,71,10)
}
for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)
print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

param_test = {
 # 'max_features':range(1,20,2)
'max_features':["auto","sqrt","log2"]
}
for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)
print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

print("--------------------第三步-----------------------")
#第三步：调节子样本比例来降低learning rate
param_test = {
 'subsample':[i/10.0 for i in range(6,10)]
}
for key,value in gsearch1.best_params_.items():
    print(key,value)
    param_test[key]=get_param(key,value)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=True,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

print("--------------------第四步-----------------------")
#第六步：降低学习速率 learning_rate =0.01, n_estimators=5000,
param_test = {
 'learning_rate':[i/100.0 for i in range(1,10)]
}

for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)

print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)


param_test = {
 # 'n_estimators':range(200,3000,50)
 'n_estimators':range(30,100,20)
}

for key,value in gsearch1.best_params_.items():
    param_test[key]=get_param(key,value)

print(param_test)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test,scoring=score_proba,n_jobs=-1,iid=False,cv=5)
gsearch1.fit(train,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)