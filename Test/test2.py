# from sklearn.datasets import load_iris
# iris = load_iris()
# # 模型(也可用单个决策树)
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=10)
# # 训练
# model.fit(iris.data, iris.target)
# # 提取一个决策树
# estimator = model.estimators_[5]
# from sklearn.tree import export_graphviz
# # 导出为dot 文件
# export_graphviz(estimator, out_file='tree.dot',
#  feature_names = iris.feature_names,
#  class_names = iris.target_names,
#  rounded = True, proportion = False,
#  precision = 2, filled = True)
# # 用系统命令转为PNG文件(需要 Graphviz)
# from subprocess import call
# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
# # 在jupyter notebook中展示
# # from IPython.display import Image
# # Image(filename = 'tree.png')


#-*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus

# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 训练模型，限制树的最大深度4
clf = RandomForestClassifier(max_depth=4)
#拟合模型
clf.fit(X, y)

Estimators = clf.estimators_
for index, model in enumerate(Estimators):
    filename = 'iris_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph