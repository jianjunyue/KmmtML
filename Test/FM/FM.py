import numpy as np
import xlearn as xl
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


path_test="/Users/jianjun.yue/PycharmGItHub/data/titanic/test_pre.csv"
path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data=pd.read_csv(path)
data_test=pd.read_csv(path_test)
print("--------------RandomForestClassifier---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare_scaler","Embarked","NameLength"]
train=data[predictors]
X=train
y=data["Survived"]
X_submission=data_test[predictors]
print(X_submission.head())
print("---------------------------")
print(train.head())

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=1)
#
# # Load dataset
# iris_data = load_iris()
# X = iris_data['data']
# y = (iris_data['target'] == 2)
#
# X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# param:
#  0. binary classification
#  1. model scale: 0.1
#  2. epoch number: 10 (auto early-stop)
#  3. learning rate: 0.1
#  4. regular lambda: 1.0
#  5. use sgd optimization method
# linear_model = xl.LRModel(task='binary', init=0.1,
#                           epoch=10, lr=0.1,
#                           reg_lambda=1.0, opt='sgd')
param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc'}
model=xl.FMModel()



model.fit(X_train, y_train)
preds =model.predict(X_test)
print("FMModel ROC AUC:%.3f" % roc_auc_score(y_true=y_test,y_score=preds))
print("FMModel accuracy_scorer:%.3f" % accuracy_score(y_true=y_test,y_pred=preds))



# Start to train
model.fit(X_train, y_train,
                 eval_set=[X_test, y_test],
                 is_lock_free=False)

y_submission=model.predict(X_submission)

y_pred=pd.DataFrame()
y_pred["PassengerId"]=data_test["PassengerId"]
y_pred["Survived"]=y_submission
y_pred["Survived"]=y_pred["Survived"].apply(lambda x: int(x))
y_pred.to_csv('/Users/jianjun.yue/KmmtML/data/kaggle/titanic/tensorflow/xlearn_submission.csv',index=None)