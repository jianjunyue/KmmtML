import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

path_test="/Users/jianjun.yue/KmmtML/data/kaggle/titanic/test.csv"
path="/Users/jianjun.yue/KmmtML/data/kaggle/titanic/train.csv"
#读取训练数据
data = pd.read_csv(path)
#查看数据情况
data.info()

#将Sex列数据转换为1或0
data['Sex'] = data['Sex'].apply(lambda s : 1 if s == 'male' else 0)
#缺失字段填充为0
data = data.fillna(0)
#选择以下特征用于分类
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()

#两种分类分别是幸存和死亡，‘Survived’字段是其中一种分类的标签
#新增加'Deceased'字段表示第二种分类的标签，取值为'Survived'取非
data['Deceased'] = data['Survived'].apply(lambda s : int(not s))
dataset_Y = data[['Deceased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()
#在训练数据中选择20%数据用来进行测试
X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=1)

print(X_train)
print("-----------------------------")
print(y_train)

#声明输入数据占位符
#shape参数的第一个元素为None,表示可以同时放入任意条记录，每条记录都有6个特征
X = tf.placeholder(tf.float32, shape=[None, 6])
y = tf.placeholder(tf.float32, shape=[None, 2])

#声明参数变量权重W和bias
W = tf.Variable(tf.random_normal([6, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')

#构造前向传播计算图
y_pred = tf.nn.softmax(tf.matmul(X, W) + bias)

#代价函数
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

#加入优化算法：随机梯度下降算法
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# 存档入口
saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 以下为训练迭代，迭代100轮
    for epoch in range(100):
        total_loss = 0.
        for i in range(len(X_train)):
            feed = {X: [X_train[i]], y: [y_train[i]]}
            # 通过session.run接口触发执行
            _, loss = sess.run([train_op, cost], feed_dict=feed)
            total_loss += loss
        print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
    print('Training complete!')

    # 评估准确率
    pred = sess.run(y_pred, feed_dict={X: X_val})
    correct = np.equal(np.argmax(pred, 1), np.argmax(y_val, 1))
    accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set: %.9f" % accuracy)

    save_path = saver.save(sess, "/Users/jianjun.yue/KmmtML/data/kaggle/titanic/tensorflow/model.ckpt")

#读入测试数据集并完成预处理,
testdata = pd.read_csv('/Users/jianjun.yue/KmmtML/data/kaggle/titanic/test.csv')
testdata = testdata.fillna(0)

testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s== 'male' else 0)
X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

with tf.Session() as sess2:
    tf.global_variables_initializer().run()
    #加载模型存档
    saver.restore(sess2, save_path)
    #正向传播计算
    predictions = np.argmax(sess2.run(y_pred, feed_dict={X:X_test}), 1)

    #构建提交结果的数据结构，并将结果存储为csv文件
    submission = pd.DataFrame({
        "PassengerId": testdata["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("/Users/jianjun.yue/KmmtML/data/kaggle/titanic/tensorflow/titanic_submission.csv", index=False)