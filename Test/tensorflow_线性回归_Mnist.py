import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(type(train_images))
print("训练集shape：")
print(train_images.shape)
print(train_images[0])
print("训练集标签shape：")
print(train_labels.shape)
print(train_labels[0])

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# image=train_images[1,:].reshape(28,28)
# plt.figure()
# plt.imshow(image)
# plt.show()

def batch_data():
    random= np.random.permutation(200)
    arr=random #random[0:100]
    train_data=train_images[arr]
    train_lable=  train_labels[arr]
    print(train_data)
    print(train_lable)
    return train_data,train_lable


#采样
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

W1 = tf.Variable(tf.zeros([784]), name="Weight")
B1 = tf.Variable(tf.zeros([1]), name="Bias")

X=tf.placeholder(tf.float32,shape=[None,784])
y=tf.placeholder(tf.float32,shape=[None,1])
# y_pred=tf.nn.softmax(tf.matmul(X,W1)+B1)
y_pred = tf.add(tf.multiply(X,W1), B1)

loss=tf.reduce_mean(tf.square(y-y_pred))
train_step=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

sess.run(train_step, feed_dict={X: train_images, y:train_labels})
curr_a, curr_b, curr_loss = sess.run([W1, B1, loss], {X: train_images, y: train_labels})
print("W= %s b: %s loss: %s" % (curr_a, curr_b, curr_loss))

# for step in range(2000):
#     batch_xs, batch_ys = batch_data()
#     sess.run(train_step, feed_dict={X: batch_xs, y:batch_ys})
#     if step % 100 == 0:
#         # correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y, 1))
#         # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         # score= sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
#         # print("score =", sess.run(accuracy))
#
#         curr_a, curr_b, curr_loss = sess.run([W1, B1, loss], {X: train_images, y: train_labels})
#         print("W= %s b: %s loss: %s" % (curr_a, curr_b, curr_loss))

# for i in range(trainIter):
#     batch=mnist
#     batchInput=batch[0]
#     batchLabels=batch[1]
#     _,train_loss= sess.run([opt,loss],feed_dict={X:batchInput,y:batchLabels})
#     if i%1000==0:
#         train_acc=ac
