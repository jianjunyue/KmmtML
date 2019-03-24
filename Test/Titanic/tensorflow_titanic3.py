import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

path_test="/Users/jianjun.yue/PycharmGItHub/data/titanic/test_pre.csv"
path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data=pd.read_csv(path)
data_test=pd.read_csv(path_test)
print("--------------tensorflow---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare_scaler","Embarked","NameLength"]
train=data[predictors]
X=train
y=data["Survived"]

#选择以下特征用于分类
dataset_X = data[predictors]
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
#shape参数的第一个元素为None,表示可以同时放入任意条记录，每条记录都有8个特征
X = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 2])

#声明参数变量权重W和bias
W = tf.Variable(tf.random_normal([8, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')

#构造前向传播计算图
y_pred = tf.nn.softmax(tf.matmul(X, W) + bias)

#代价函数
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

#加入优化算法：随机梯度下降算法
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

n_epochs = 100
batch_size = 10
m, n = X_train.shape
n_batches = int(np.ceil(m / batch_size))  # ceil() 方法返回 x 的值上限 - 不小于 x 的最小整数。
def fetch_batch(batch_size):
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = X_train[indices]  # not shown
    y_batch = y_train[indices]  # not shown
    return X_batch, y_batch

# 存档入口
saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 以下为训练迭代，迭代100轮
    for epoch in range(100):
        total_loss = 0.
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(batch_size)
            feed = {X: X_batch, y: y_batch}
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

    # 正向传播计算
    predictions = np.argmax(sess.run(y_pred, feed_dict={X: X_val}), 1)
    y_test=np.argmax(y_val, 1)
    print("tensorflow ROC AUC:%.3f" % roc_auc_score(y_true=y_test, y_score=predictions))
    print("tensorflow accuracy_scorer:%.3f" % accuracy_score(y_true=y_test, y_pred=predictions))

    save_path = saver.save(sess, "/Users/jianjun.yue/KmmtML/data/kaggle/titanic/tensorflow/model.ckpt")

#读入测试数据集并完成预处理,
testdata = pd.read_csv(path_test)
X_test = testdata[predictors]

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