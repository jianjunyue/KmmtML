from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

# !pip install tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow import feature_column
# from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/feature_columns.ipynb
# tf.keras.layers.DenseFeatures()
print(tf.__version__)

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
print(dataframe.head())
dataframe.to_csv("/Users/jianjun.yue/KmmtML/data/tensorflow/feature_columns/heart.csv")