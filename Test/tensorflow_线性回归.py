import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential

# 产生训练数据集
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# 产生测试样本
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

set=[]
for i in range(1000):
    x1=np.random.normal(0.0,0.55)
    y1=x1*0.1+0.3+np.random.normal(0.0,0.03)
    set.append([x1,y1])
train_X=[v[0] for v in set]
train_Y=[v[1] for v in set]
print(type(train_X))
print(train_X)


# 展示原始数据分布
# plt.plot(train_X, train_Y, 'ro', label='Original Train Points')
# plt.plot(test_X, test_Y, 'b*', label='Original Test Points')
# plt.legend()
# plt.show()

# 回归模型的权重和偏置：np.random.randn()返回一个标准正态分布随机数
W = tf.Variable(np.random.randn(), name="Weight")
b = tf.Variable(np.random.randn(), name="Bias")
# inference: 创建一个线性模型：y = wx + b
y = tf.add(tf.multiply(train_X,W), b)
# y=W * train_X + b

loss=tf.reduce_mean(tf.square(y-train_Y))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

print("W =",sess.run(W),"b =",sess.run(b),"loss =",sess.run(loss))

for step in range(20):
    sess.run(train)
    print("W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss))

# temp_y = sess.run( tf.add(tf.multiply(train_X,W), b))
# plt.scatter(train_X,train_Y,c="r")
# plt.plot(train_X,temp_y)
# plt.show()
