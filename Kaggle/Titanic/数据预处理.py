import pandas as pd
import numpy as np

# titanic=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/titanic/train.csv")
titanic=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/titanic/test.csv")

# print(titanic.head())
# print(titanic.describe())
age=int(titanic["Age"].median())
print(age)
#数据预处理
titanic["Age"]=titanic["Age"].fillna(age)
titanic["Age"]=titanic["Age"].apply(lambda x:int(x))
# print(titanic.describe())
# print(titanic.describe())

# print(titanic["Sex"].unique())

titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1

print(titanic["Sex"].unique())

# print(titanic["Embarked"].unique())
titanic["Embarked"]=titanic["Embarked"].fillna("S")
# print(titanic["Embarked"].unique())
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2

#特征工程
titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]
titanic["NameLength"]=titanic["Name"].apply(lambda x:len(x))
print("------------------------titanic.isnull().values==True---------------------------------------")
print(titanic[titanic.isnull().values==True])
Fare=int(titanic["Fare"].median())
titanic["Fare"]=titanic["Fare"].fillna(Fare)
titanic["Fare"]=titanic["Fare"].apply(lambda x:float(x))
import re
def getTitle(name):
    title_search=re.search("([A-Za-z]+)\.",name)
    if title_search:
        return title_search.group(1)
    return ""
# print(getTitle("Braund, Mr. Owen Harris"))
titles=titanic["Name"].apply(getTitle)
# print(pd.value_counts(titles))

title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Major":7,"Col":8,"Mlle":9,"Sir":10,"Lady":11,"Ms":12,"Lady":13,"Jonkheer":14,"Mme":15,"Capt":16,"Don":17,"Countess":18,"Dona":19}
for k,v in title_mapping.items():
    titles[titles==k]=v
# print(pd.value_counts(titles))
titanic["Title"]=titles
# print(titanic)
titanic.drop(["Name","Cabin","Ticket"],axis=1,inplace=True)
print("----------------------titanic.head()---------------------------------")
print(titanic.head(10))
# titanic.to_csv("train.csv",index=False)
titanic.to_csv("test.csv",index=False)
print("----------------------titanic.head()---------------------------------")






















