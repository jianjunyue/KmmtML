import tensorflow as tf
mnist = tf.keras.datasets.mnist
#
# print(mnist)


# print(tf.version)
print(tf.__version__)
print(tf.keras.__version__)

def loss(predicted_y, desired_y):
  return tf.reduce_mean(tf.square(predicted_y - desired_y))