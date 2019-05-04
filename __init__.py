import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

df=pd.DataFrame()
value1=["1","2","3"]
value2=[1,2,3]
value3=[30,40,50]
value4=[100,200,300]
df["id"]=value1
df["gender"]=value2
df["age"]=value3
df["value"]=value4
print(df)
print("------")

X = df[['gender', 'age', 'value']]

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_ploly = poly.fit_transform(X)
X_ploly_df = pd.DataFrame(X_ploly, columns=poly.get_feature_names())
print(X_ploly_df.head())




