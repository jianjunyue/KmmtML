import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

path_test="/Users/jianjun.yue/PycharmGItHub/data/titanic/test_pre.csv"
path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data=pd.read_csv(path)
data_test=pd.read_csv(path_test)
print("--------------RandomForestClassifier---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare_scaler","Embarked","NameLength"]
train=data[predictors]
X=train
y=data["Survived"]
X_submission=data_test[predictors]
print(X_submission.head())
# print(X_submission.describe())
print("---------------------------")
print(train.head())
# print(train.describe())

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=1)

# 读取文件中的数据
def read_csv(batch_size, file_name, record_defaults):
    # file_path = os.getcwd()+'\\'+file_name
    # print (file_path)
    filename_queue = tf.train.string_input_producer([file_name])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv会将字符串(文本行)转换到具有指定默认值的由张量列构成的元组中
    # 它还会为每一列设置数据类型
    decode = tf.decode_csv(value, record_defaults=record_defaults)

    # 实际上读取一个文件，并加载一个张量中的batch_size行
    return tf.train.shuffle_batch(decode,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


# 参数初始化
W = tf.Variable(tf.zeros([5, 1], name="weights"))
b = tf.Variable(0., name="bias")


def combine_inputs(X):
    L = tf.matmul(X, W) + b
    return L


def inference(X):
    return tf.sigmoid(combine_inputs(X))


# 使用交叉熵
def loss(X, Y):
    L =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))
    return L


def inputs():
    record_defaults = [[0.], [0.], [0.], [""], [""], [0.], [0.], [0.], [""], [0.], [""], [""]]
    passenger_id, surived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(891,r"/Users/jianjun.yue/KmmtML/data/kaggle/titanic/train.csv",record_defaults)
    # 转换属性数据
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))
    gender = tf.to_float(tf.equal(sex, ["female"]))

    # 最终将所有特征排列在一个矩阵中，然后对该矩阵进行转置，使其每行对应一个样本，每列对应一种特征
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    surived = tf.reshape(surived, [891, 1])
    return features, surived


def train(total_loss):
    learning_rate = 0.008
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    predicated = tf.cast(inference(X) > 0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicated, Y), tf.float32))))


################启动会话主程序##############
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    # print (total_loss)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # print (threads)

    # 实际训练的迭代次数
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # 查看损失在训练过程中的递减情况
        if step % 100 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()