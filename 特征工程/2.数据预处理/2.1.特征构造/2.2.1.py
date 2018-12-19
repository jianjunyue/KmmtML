from Data.titanic.TitanicData import TitanicData
from Utils.Util import Util
from BuildFeature.BuildBase import BuildBase
from scipy import stats

# 2.2缺失值处理
# 2.2.1 离散型一般用众数，连续型用中位数或者均值。


data_original,data_test_original=TitanicData.data_original()
# print(data_original)
data=data_original
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# keyName="Pclass"
# keyName="Name"
# keyName="Sex"
keyName="Age"
# keyName="SibSp"
# keyName="Parch"
# keyName="Ticket"
# keyName="Fare"
# keyName="Cabin"
# keyName="Embarked"
Util.data_info_print(data_original,data_test_original,keyName)

print("---------------------------------------------")

age_mode=BuildBase.mode(data_original,"Age")
# 众数，连续型用中位数或者均值

age=data_original["Age"].mode()[0]
print("众数:"+str(age)+".")
#中位数
age=data_original["Age"].median()
print("中位数"+str(age))
#均值
age=data_original["Age"].mean()
print(age)

print("-----BuildBase----")
print(BuildBase.mean(data_original,"Age"))
print(BuildBase.mode(data_original,"Age"))
print(type(BuildBase.mode(data_original,"Age")))