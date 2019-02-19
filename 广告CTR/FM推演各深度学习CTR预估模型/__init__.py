
import numpy as np
import tensorflow as tf


a=tf.constant([1.0,2.0,3.0],name="testA")
b=tf.constant([5.0,6.0,8.0],name="testB")
result=tf.add(a,b,name="testAandB")
print(result)

result1=a+b
print(result1)

sess=tf.Session()
run=sess.run(result)
print(run)
sess.close()