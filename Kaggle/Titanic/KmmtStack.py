# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# https://blog.csdn.net/qq_36330643/article/details/78576503

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
    clf = LogisticRegression(penalty="l1", tol=1e-6, C=1.0, random_state=1, n_jobs=-1)
    clf.fit(dataset_blend_train, train_y)
    prediction = clf.predict(dataset_blend_test)
    return prediction



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
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, y, test_size=0.1, random_state=111)


# 5个一级分类器
clfs = [SVC(C = 3, kernel="rbf"),
            RandomForestClassifier(n_estimators=100, max_features="log2", max_depth=10, min_samples_leaf=1, bootstrap=True, n_jobs=-1, random_state=1),
            KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=100, criterion="gini", max_features="log2", max_depth=10, min_samples_split=2, min_samples_leaf=1,bootstrap=True, n_jobs=-1, random_state=1)]

# (train_x, train_y, test) = load_data()
n_folds=5
prediction = stack(clfs,X_train, y_train, X_test,n_folds)
print(prediction)