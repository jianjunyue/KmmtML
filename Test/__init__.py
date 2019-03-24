import tensorflow as tf
import numpy as np

test_count = 10         #数据集数量
param_count = 5         #变量数
x_train = np.floor(10 * np.random.random([test_count,param_count]),dtype=np.float32)
y_train = np.floor(10 * np.random.random([test_count,1]),dtype=np.float32)
print(x_train)
print(y_train)

W = tf.Variable(tf.zeros([1,5]), tf.float32)
b = tf.Variable(tf.zeros([1,1]), tf.float32)

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

y_pred = tf.add(tf.multiply(X,W), b)

loss=tf.reduce_sum(tf.square(y-y_pred))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(11):
    sess.run(train, {X: x_train, y: y_train})

curr_a, curr_b, curr_loss = sess.run([W, b, loss], {X: x_train, y: y_train})
print("W= %s b: %s loss: %s" % (curr_a, curr_b, curr_loss))





