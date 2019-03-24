#https://github.com/coreylynch/pyFM
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from pyfm import pylibfm

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,n_features=100, n_clusters_per_class=1)
data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]

path = "/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data_df = pd.read_csv(path)
print("--------------RandomForestClassifier---------------")
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare_scaler", "Embarked", "NameLength"]
train = data_df[predictors]
y = data_df["Survived"]
data = [{v: k for k, v in dict(zip(i, range(len(i)))).items()} for i in train.values.tolist()]

# X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=1)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

v = DictVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.0001, learning_rate_schedule="optimal")

fm.fit(X_train,y_train)

y_pred=fm.predict(X_test)
print(y_pred)

# Evaluate
from sklearn.metrics import log_loss
print("Validation log loss: %.4f" % log_loss(y_test,y_pred))