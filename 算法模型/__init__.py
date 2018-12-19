
import pandas as pd
import numpy as np
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())
listdata = np.array([[1, 2], [2, 3], [3, 5]])
print(type(listdata))
listdata = np.array([[2], [3], [5]])
print(type(listdata))
print(type(listdata))

df=pd.DataFrame()
value1=["1","2","3"]
value2=["mall","room","group"]
df["id"]=value1
df["gender"]=value2
print(type(df["id"]))

a=[1,1,1]
print(type(a))
b=np.array([1])
print(type(b))
