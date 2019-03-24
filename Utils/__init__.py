import numpy as np
import pandas as pd
from KmmtML.Utils.DataHelper import DataHelper

def CSV2Libsvm():
    path_test = "/Users/jianjun.yue/PycharmGItHub/data/titanic/test_pre.csv"
    path = "/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
    data = pd.read_csv(path)
    data_test = pd.read_csv(path_test)
    print("--------------RandomForestClassifier---------------")
    predictors = ["Survived","Pclass", "Sex", "Age", "SibSp", "Parch", "Fare_scaler", "Embarked", "NameLength"]
    train = data[predictors]
    y = data["Survived"]
    print("---------------------------")
    print(train.head())
    print(len(train.values[0]))
    # train["Survived"]=train["Survived"].astype('category')

    libsvm=[]
    for row_array in train.values:
        tempsvm = ""
        for i in range(len(row_array)):
            if(i==0):
                tempsvm=str(row_array[i])+" "
            else :
                if(int(row_array[i])!=0):
                    tempsvm += str(i)+":"+str(row_array[i]) + " "
        libsvm.append(tempsvm.strip())

    dt=DataHelper.list_to_df(libsvm)
    dt.to_csv("/Users/jianjun.yue/KmmtML/KmmtML/Test/FM/submission.csv",index=None,header=None,columns=None)






CSV2Libsvm()