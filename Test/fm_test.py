from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# https://blog.csdn.net/klxmm68/article/details/82792091

train = [
	{"user": "1", "item": "5", "age": 19},
	{"user": "2", "item": "43", "age": 33},
	{"user": "3", "item": "20", "age": 55},
	{"user": "4", "item": "10", "age": 20},
]

v = DictVectorizer()
X = v.fit_transform(train)

print(X.toarray())
y = np.repeat(1.0,X.shape[0])
fm = pylibfm.FM()
fm.fit(X, y)
out=fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
print(out)