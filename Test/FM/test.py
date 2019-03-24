import xlearn as xl
from KmmtML.Utils.Util import Util
import numpy as np
import pandas as pd

# Training task
ffm_model = xl.create_ffm()                # Use field-aware factorization machine (ffm)
ffm_model.setTrain("./small_train.txt")    # Set the path of training data

fm_model = xl.create_fm()
fm_model.fit()
# parameter:
#  0. task: binary classification
#  1. learning rate : 0.2
#  2. regular lambda : 0.002
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}

path = "/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data = pd.read_csv(path)
print("--------------RandomForestClassifier---------------")
predictors = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare_scaler", "Embarked", "NameLength"]
train = data[predictors]

dt= Util.CSV2Libsvm(train)

print(dt)

xl.FMModel(task='binary', init=0.1,
                          epoch=10, lr=0.1,
                          reg_lambda=1.0, opt='sgd')
fm=xl.FMModel()
fm.fit()