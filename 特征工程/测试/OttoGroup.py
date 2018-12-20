from KmmtML.Data.FileData import FileData
from KmmtML.Utils.FEHelper import FEHelper
from KmmtML.BuildFeature.BuildFeatureBase import BuildFeatureBase

from sklearn.ensemble import RandomForestClassifier

dir_name="otto_group"
file_name="train"
test_file_name="test"
data=FileData.getData(dir_name,file_name)
data_test=FileData.getData(dir_name,test_file_name)
# print(train.head())
# FEHelper.data_info(test)

predictors=[]
for column in data.columns:
    if column[:5]=="feat_":
        predictors.append(column)

# print(predictors)
train=data[predictors]
# gender_mapping={"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}
# gender_mapping=BuildFeatureBase.labelencoder_mapping(data,"target")
df=BuildFeatureBase.labelencoder_df(data,"target")
data["target"]=df["target"]
y=data["target"]
keyName="target"
print(y)
# print(df["target"])
# train["target"]=data["feat_1"]
# y=train["target"]

# model = RandomForestClassifier()
# model.fit()