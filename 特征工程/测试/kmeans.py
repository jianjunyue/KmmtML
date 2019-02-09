from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.datasets import make_moons
import numpy as np
from KmmtML.Algorithm.KMeansFeaturizer import KMeansFeaturizer

# training_data, training_labels = make_moons(n_samples=2000, noise=0.2)

training_data=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[111],[20],[22],[1],[3],[5],[6],[8],[9],[5],[22],[1],[3],[5],[6],[8],[9],[5],[2],[1],[3],[5],[6],[8],[9],[5],[12]]
test=[[2],[8],[10],[30],[60],[100]]
km_model = KMeansFeaturizer(k=5, target_scale=10,random_state=7).fit(training_data, None)
# kmf_no_hint = KMeansFeaturizer(k=100, target_scale=0).fit(training_data, training_labels)
# ky=kmf_hint.transform(test)

clusters =  km_model.predict(test)
print(clusters)
print("-----------------------------------------")
print(training_data)
print("-----------------------------------------")
# print(ky)