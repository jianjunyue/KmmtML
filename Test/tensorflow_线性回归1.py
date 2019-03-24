import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential

# 产生训练数据集
x_train = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
y_train = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

W = tf.Variable(tf.zeros([1]), tf.float32)
b = tf.Variable(tf.zeros([1]), tf.float32)

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

y_pred = tf.add(tf.multiply(X,W), b)

loss=tf.reduce_mean(tf.square(y-y_pred))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {X: x_train, y: y_train})

curr_a, curr_b, curr_loss = sess.run([W, b, loss], {X: x_train, y: y_train})
print("W= %s b: %s loss: %s" % (curr_a, curr_b, curr_loss))













