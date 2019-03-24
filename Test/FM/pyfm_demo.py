#https://github.com/coreylynch/pyFM

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm

train = [
	{"user": "1", "item": "5", "age": 19},
	{"user": "2", "item": "43", "age": 33},
	{"user": "3", "item": "20", "age": 55},
	{"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
print(type(X))
print(print(v.get_feature_names()))
print(X)
# print(X.toarray())
# [[ 19.   0.   0.   0.   1.   1.   0.   0.   0.]
#  [ 33.   0.   0.   1.   0.   0.   1.   0.   0.]
#  [ 55.   0.   1.   0.   0.   0.   0.   1.   0.]
#  [ 20.   1.   0.   0.   0.   0.   0.   0.   1.]]
y = np.repeat(1.0,X.shape[0])
print(y)
fm = pylibfm.FM()
fm.fit(X,y)
pred=fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
print(pred)
print(v.transform({"user": "10", "item": "10", "age": 24}))
print(v.transform({"user": "10", "item": "10", "age": 24}).toarray())