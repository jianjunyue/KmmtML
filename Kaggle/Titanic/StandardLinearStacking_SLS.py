"""
SLS ensemble classifier begins here
"""
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import sls_logistic_reg_classifier as sls
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

path="train.csv"
data=pd.read_csv(path)
path_test="test.csv"
data_test=pd.read_csv(path_test)

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]
train=data[predictors]
y=data["Survived"]
test=data_test[predictors]
submission=pd.DataFrame()
submission["PassengerId"]=data_test["PassengerId"]

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.4, random_state=1)

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

model2 = RandomForestClassifier(
    n_estimators=30,
    max_depth=8,
    min_samples_split=50,
    min_samples_leaf=10,
    max_features=9,
    oob_score=True,
    random_state=10,
    n_jobs=-1)

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
)


#Apply standarized scaling to the SVM and KNN models
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', model1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', model3]])

#Now implement the SLS classifier
sls_clf = sls.SLS_Classifier(classifiers=[model1, model2, model3])

#Fit the component models to the training set
sls_clf.fit(x_train, y_train)
#Predict classifications of training set observations for each component model
y_train_pred = sls_clf.predict(x_train)
#Predict classifications of test set observations for each component model
y_test_pred = sls_clf.predict(x_test)

#Fit a logistic regression to the predicted y values for each model
pipe_lr = Pipeline([('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(y_train_pred, y_train)
#Predict the y values of the test set using the fitted SLS ensemble model (the prediction is made with the composite predictions of the component models)
y_pred = pipe_lr.predict(y_test_pred)

#View the test accuracy of the SLS ensemble model
print('Test Accuracy of SLS ensemble model: %.3f' % pipe_lr.score(y_test_pred, y_test))

#测试效果
y_test_sub = sls_clf.predict(test)
y_pred_sub = pipe_lr.predict(y_test_sub)
submission["Survived"]=y_pred_sub
submission.to_csv('data/SLS.csv',index=None)

# #Build a confusion matrix
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
#
# confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
# plt.xlabel('predicted values')
# plt.ylabel('true values')
# plt.title('Confusion Matrix: SLS Ensemble Classifier')
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# plt.show()