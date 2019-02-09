from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from KmmtML.BuildFeature.BuildFeatureBase import BuildFeatureBase
from KmmtML.Data.FileData import FileData
import matplotlib.pyplot as plt

# 2.12 非正态分布转正态分布
# 非正态分布转正太分布（log），平方，立方，根号…（但任何针对单独特征列的单调变换（如对数）：不适用于决策树类算法。对于决策树而言，X 、X^3 、X^5 之间没有差异， |X| 、 X^2 、 X^4 之间没有差异，除非发生了舍入误差。）


train=FileData.getData("titanic/","train")

x=["a1","a2","a3","a4","a5","a6","a7","a8"]
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,111,20,22,1,3,5,6,8,9,5,22,1,3,5,6,8,9,5,2,1,3,5,6,8,9,5,12]
df = pd.DataFrame(columns=['key', 'key_name', 'count'])
df["key"]=x
# df["key_name"]=x
# df["count"]=y
# df["count1"]=y

print(df.head(10))

plt.figure(figsize=(8,6))
#
# plt.plot(range(len(df["key"].values)),  df["count"].values)
# plt.scatter(range(len(df["key"].values)),  df["count1"].values,s=30,c='blue',marker='x',alpha=0.5,label='C2')
# plt.xticks(range(len(df["key"].values)),df["key_name"].values) #给X轴赋值名称
# # plt.xticks(range(len(df["key"].values)),df["key_name"].values)
# plt.legend(loc='upper right')
# plt.title("test title")
# plt.xlabel('test X');
# plt.ylabel('test Y');
# plt.show()

#hist x轴表示实际值，y轴表示值的个数,bins块数
# plt.hist(np.log1p(df["key"]), bins=5)
# plt.hist(df["key"], bins=8)
plt.hist(np.log1p(train["Fare"]), bins=100)
plt.title("test title")
plt.xlabel('test X');
plt.ylabel('test Y');
plt.show()

