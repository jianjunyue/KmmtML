from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from Algorithm.FeaturePreprocess.FEMinMaxScaler import FEMinMaxScaler

# https://wenku.baidu.com/view/ba53ac32f02d2af90242a8956bec0975f465a4bb.html
K=range(2,8)
n_samples = 1500
random_state = 170
# X, y = make_blobs(n_samples=n_samples, random_state=random_state)
# print(X)
X=[[1],[2],[3],[4],[8],[9],[10],[11]]

X=FEMinMaxScaler.fit_transform(X)
SSE=[]
# for k in K:
#     kmeans = KMeans(n_clusters=k)
#     model=kmeans.fit(X)
#     labels=model.labels_
#     print(labels)
#     print(model.inertia_)
#     s=metrics.silhouette_score(X,labels)
#     SSE.append(s)

for k in K:
    kmeans = KMeans(n_clusters=k)
    model=kmeans.fit(X)
    labels=model.labels_
    print(labels)
    print(model.inertia_)
    # s=metrics.silhouette_score(X,labels)
    SSE.append(model.inertia_)

plt.xlabel("k")
plt.ylabel("SSE")
plt.plot(K,SSE,"*-")
plt.show()


print(SSE)

# y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
# print(y_pred)