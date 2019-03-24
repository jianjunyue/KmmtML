import numpy as np
from sklearn.metrics import accuracy_score

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
score = y_true == y_pred
print(score)

test= 1 if y_true == y_pred else 0
print(test)

import tensorflow as tf

# y_true = tf.constant([0, 1, 0, 1, 1, 1, 0, 1, 0])
# y_pred = tf.constant([0, 1, 1, 1, 0, 1, 0, 1, 1])
#若数组为多维数据，则需要通过reshape来转换为1维数据

y_true = np.array([[0, 1],
                  [1, 0],
                  [1, 0]])
y_pred = np.array([[1, 0],
                  [0, 1],
                  [1, 0]])

###############

# y_true = np.array([[0, 1, 1],
#                   [0, 1, 0]])
# y_pred = np.array([[1, 1, 1],
#                   [0, 0, 1]])
y_true = np.reshape(y_true, [-1]) #[0 1 1 0 1 0]
y_pred = np.reshape(y_pred, [-1]) #[1 1 1 0 0 1]



TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.int32))
# TP = tf.reduce_sum(tf.multiply(y_true, y_pred))
FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), tf.int32))
FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), tf.int32))
TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 0)), tf.int32))

# A = (TP + FN) / (P + N)
# P = TP / (TP + FP)
# R = TP / (TP + FN)
# F1 = 2 * P * R / (P + R)

sess = tf.Session()
print(sess.run(tf.size(y_true)))
print(sess.run(TP))
print(sess.run(FP))
print(sess.run(FN))
print(sess.run(TN))

test=(TP+TN) / (tf.size(y_true))

print(sess.run(test))
