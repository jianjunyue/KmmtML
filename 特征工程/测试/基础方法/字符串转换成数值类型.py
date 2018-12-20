import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from BuildFeature.BuildFeatureBase import BuildFeatureBase


#对于有序特征，用数值编码（可区分特征大小关系）；对于无序特征，用独热编码
df=pd.DataFrame()
value=["mall","room","group","data"]
df["gender"]=value
print(df)
print("---------LabelEncoder---------")
# lb = preprocessing.LabelBinarizer()
# gender 改为 0-1 数值
# df["gender"]= lb.fit_transform(df['gender'])

#标签编码（Label encoding）
le = preprocessing.LabelEncoder()
df["gender_1"]= le.fit_transform(df['gender'])
print(df)

gender_mapping={lable:index for index,lable in enumerate(le.classes_)}
print(gender_mapping)


print("---------map 实现---------")
# gender_mapping={"mall":1,"room":2,"group":3,"data":4}
df["gender_2"]=df["gender"].map(gender_mapping)
print(df)


print("---------BuildBase 实现---------")

labelencoder_mapping=BuildFeatureBase.labelencoder_mapping(df,"gender")
print(labelencoder_mapping)

labelencoder_df=BuildFeatureBase.labelencoder_df(df,"gender",labelencoder_mapping)
print(labelencoder_df)


labelencoder_df=BuildFeatureBase.labelencoder_df(df,"gender")
print(labelencoder_df)