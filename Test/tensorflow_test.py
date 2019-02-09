import tensorflow as tf
import numpy as np
from keras.models import Sequential

X = np.array([[2, 3], [4, 5], [6, 7]])
y = np.array(["cat", "dog", "fox"])
dataset = tf.data.Dataset.from_tensor_slices((X, y))
print(dataset)


for item_x, item_y in dataset:
    print(item_x.numpy(), item_y.numpy())