import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)


mnist = tf.keras.datasets.mnist


data = mnist.load_data()

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# 计算图的输入
X = tf.placeholder(tf.float32,[None,784]) # mnist图片尺寸为28*28=784
Y = tf.placeholder(tf.float32,[None,10]) # 0-9共9个数字，10分类问题
# 模型权重
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 构建模型
pred = tf.nn.softmax(tf.matmul(X,W)+b)
# crossentroy
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=1))
# SGD
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# 初始化
init = tf.global_variables_initializer()

tf.Session().run()