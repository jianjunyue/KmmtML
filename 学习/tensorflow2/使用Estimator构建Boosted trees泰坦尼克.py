import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# https://zhuanlan.zhihu.com/p/61400276


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
print(X_submission.describe())

print("---------------------------")
print(train.head())
print(train.describe())

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=1)
print(X_train)
print("------------------------")
print(y_train)


seed = 2017
print("------------------------RandomForestClassifier-------------------------------")
rfc=RandomForestClassifier(random_state=seed)
rfc.fit(X_train, y_train)
preds =rfc.predict(X_test)
print("RandomForestClassifier ROC AUC:%.3f" % roc_auc_score(y_true=y_test,y_score=preds))
print("RandomForestClassifier accuracy_scorer:%.3f" % accuracy_score(y_true=y_test,y_pred=preds))


fc = tf.feature_column
CATEGORICAL_COLUMNS = ['Pclass', 'Sex', 'Parch', 'SibSp', 'Embarked']
NUMERIC_COLUMNS = ['Age', 'NameLength']

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab))
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = X_train[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))


# 构造输入数据.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(X_train, y_train)
eval_input_fn = make_input_fn(X_test, y_test, shuffle=False, n_epochs=1)

n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,n_batches_per_layer=n_batches)

# 训练
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(eval_input_fn)
print(pd.Series(result))