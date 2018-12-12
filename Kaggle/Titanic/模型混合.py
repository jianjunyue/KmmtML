# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from sklearn.metrics import make_scorer
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


path="train.csv"
data=pd.read_csv(path)
path_test="test.csv"
data_test=pd.read_csv(path_test)
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]
train=data[predictors]
y=data["Survived"]
test=data_test[predictors]

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=111)

model1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=200,
 max_depth=3,
 min_child_weight=2,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=1e-05,
 objective= 'binary:logistic',
 scale_pos_weight=1,
 seed=27)

model1.fit(X_train,y_train)
y_pred=model1.predict(X_test)

accuracy_score=metrics.accuracy_score(y_pred,y_test)
print("XGBClassifier : ",accuracy_score)

model2 = RandomForestClassifier(
    n_estimators=30,
    max_depth=8,
    min_samples_split=50,
    min_samples_leaf=10,
    max_features=9,
    oob_score=True,
    random_state=10,
    n_jobs=-1)

model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)

accuracy_score=metrics.accuracy_score(y_pred,y_test)
print("RandomForestClassifier : ",accuracy_score)

model3 = GradientBoostingClassifier(
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

model3.fit(X_train,y_train)
y_pred=model3.predict(X_test)

accuracy_score=metrics.accuracy_score(y_pred,y_test)
print("GradientBoostingClassifier : ",accuracy_score)

def stack(clfs, train_x, train_y, test, n_folds):
    """ stacking
    input: train_x, train_y, test
    output: test的预测值
    clfs: 多个一级分类器
    train_x, test: 二级分类器的train_x, test
    n_folds: 多个分类器进行n_folds预测
    dataset_blend_train: 一级分类器的prediction, 二级分类器的train_x
    dataset_blend_test: 二级分类器的test
    """
    # idx = np.random.permutation(train_y.size)
    train_x = train_x.values
    train_y = train_y.values
    test = test.values
    dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)), dtype=np.int)
    dataset_blend_test = np.zeros((test.shape[0], len(clfs)), dtype=np.int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    for i, clf in enumerate(clfs):
        dataset_blend_test_j = np.zeros((test.shape[0], n_folds))  # 每个分类器的单次fold预测结果
        for j, (train_index, test_index) in enumerate(skf.split(train_x, train_y)):
            # print(train_index, test_index)
            tr_x = train_x[train_index]
            tr_y = train_y[train_index]
            clf.fit(tr_x, tr_y)
            dataset_blend_train[test_index, i] = clf.predict(train_x[test_index])
            dataset_blend_test_j[:, j] = clf.predict(test)
        dataset_blend_test[:, i] = dataset_blend_test_j.sum(axis=1) // (n_folds // 2 + 1)

    # 二级分类器进行预测
    clf = LogisticRegression(penalty="l1", tol=1e-6, C=1.0, random_state=1)
    clf.fit(dataset_blend_train, train_y)
    prediction = clf.predict(dataset_blend_test)
    return prediction

# 5个一级分类器
clfs = [model1,model2,model3]
n_folds=5
y_pred = stack(clfs,X_train, y_train, X_test,n_folds)
accuracy_score=metrics.accuracy_score(y_pred,y_test)
print("Stack : ",accuracy_score)

from sklearn.cross_validation import StratifiedKFold

def blend(clfs, X_train_, y_train_, X_test_, n_folds):
    X = X_train_.values
    y = y_train_.values
    X_submission = X_test_.values
    skf = list(StratifiedKFold(y, n_folds))
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    for j, clf in enumerate(clfs):
        # print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            # print("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            # y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    clf = LogisticRegression(penalty="l1", tol=1e-6, C=1.0, random_state=1)
    clf.fit(dataset_blend_train, y)
    # y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
    prediction = clf.predict(dataset_blend_test)
    return prediction

# 5个一级分类器
clfs = [model1,model2,model3]
n_folds=5
y_pred = blend(clfs,X_train, y_train, X_test,n_folds)
accuracy_score=metrics.accuracy_score(y_pred,y_test)
print("Blend : ",accuracy_score)
