import collections
import math
import os
import time

import tensorflow as tf

tf.app.flags.DEFINE_string("tables", "", "table info")
tf.app.flags.DEFINE_string("checkpointDir", "", "oss info")
FLAGS = tf.app.flags.FLAGS


def read_from_odps(filename):
    defaults = collections.OrderedDict([
        ("is_order", [0]),
        ("features", [""])
    ])

    def decode_line(*items):
        pairs = zip(defaults.keys(), items)
        features_dict = dict(pairs)
        # feat_index
        indexes = tf.string_split([features_dict['features']], ' ')
        feat_idx = tf.string_to_number(indexes.values, out_type=tf.int32)
        features_dict["features"] = feat_idx
        label_order = features_dict.pop("is_order")
        label_order = tf.cast(label_order, tf.float32)
        features_dict['label'] = tf.reshape(label_order, [1])
        return features_dict

    dataset = tf.data.TableRecordDataset(filename, tuple(v[0] for v in defaults.values()),
                                         selected_cols="is_order,features")
    dataset = dataset.map(decode_line, num_parallel_calls=16)
    dataset = dataset.shuffle(buffer_size=200000)
    dataset = dataset.batch(10000)
    dataset = dataset.repeat(10)
    dataset = dataset.prefetch(1)
    return dataset


def weight_variable(shape, stddev, name):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
    return initial


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def nerve_cell(in_put, w, b, keep_prob, norm, out_size, on_train):
    Wx_plus_b = tf.matmul(in_put, w) + b
    # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
    # Wx_plus_b = Wx_plus_b * scale + shift
    if norm:
        fc_mean, fc_var = tf.nn.moments(
            Wx_plus_b,
            axes=[0], )
        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = tf.cond(on_train,
                            mean_var_with_update,
                            lambda: (
                                ema.average(fc_mean),
                                ema.average(fc_var)
                            ))
        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
    return tf.nn.dropout(tf.nn.relu(Wx_plus_b), keep_prob=keep_prob)


handle = tf.placeholder(tf.string, shape=[])
data_t = read_from_odps(FLAGS.tables)
iterator = tf.data.Iterator.from_string_handle(handle, data_t.output_types, data_t.output_shapes)
next_map = iterator.get_next()
training_iterator = data_t.make_initializable_iterator()

feature = tf.placeholder_with_default(next_map['features'], [None, 500 + 365], name='feature')
label = tf.placeholder_with_default(next_map['label'], [None, 1], name='label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
is_train = tf.placeholder(tf.bool)

bias = tf.Variable(tf.random_normal([1], stddev=0.1), name='bias')

first_order_weight = tf.Variable(tf.random_normal([365 * 21 + 500 * 128 + 1, 1], stddev=0.1), name='w_linear',
                                 dtype=tf.float32)
first_order_embedding = tf.nn.embedding_lookup(first_order_weight, feature)
first_order_y = tf.reduce_sum(first_order_embedding, 1)  # None * 1

embedding_table = tf.Variable(tf.random_uniform([365 * 21 + 500 * 128 + 1, 10], minval=-1.0 / math.sqrt(500 + 363),
                                                maxval=1.0 / math.sqrt(500 + 363)), name='w_fm', dtype=tf.float32)
input_embedding = tf.nn.embedding_lookup(embedding_table, feature)  # None * field_size * embedding_size
sum_square = tf.square(tf.reduce_sum(input_embedding, 1))  # None * field_size
square_sum = tf.reduce_sum(tf.square(input_embedding), 1)  # None * field_size
second_order_y = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)  # None * 1

last_layer_neurons = (365 + 500) * 10

hidden_layers = tf.reshape(input_embedding, shape=[-1, last_layer_neurons])  # None * (field_size * embedding_size)

bias_1 = bias_variable([128], name="bias_1")
weights_1 = weight_variable([8650, 128], stddev=1 / 8650.0 / 2.0, name="weights_1")
hidden_layer_1 = nerve_cell(hidden_layers, weights_1, bias_1, keep_prob, True, 128, is_train)

bias_1_1 = bias_variable([64], name="bias_1_1")
weights_1_1 = weight_variable([128, 64], stddev=1 / 128.0 / 2.0, name="weights_1_1")
hidden_layer_1_1 = nerve_cell(hidden_layer_1, weights_1_1, bias_1_1, keep_prob, True, 64, is_train)

w_output = tf.Variable(tf.random_uniform([64, 1], minval=-1.0 / math.sqrt(64), maxval=1.0 / math.sqrt(64)),
                       name='w_output', dtype=tf.float32)
nn_output_y = tf.matmul(hidden_layer_1_1, w_output)

logits = bias + nn_output_y + first_order_y

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)) + 0.1 * (
        0.0001 * tf.nn.l2_loss(weights_1) + 0.0001 * tf.nn.l2_loss(weights_1_1) +
        0.0001 * tf.nn.l2_loss(first_order_weight))

loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))

predict = tf.nn.sigmoid(logits, name='predict')

auc = tf.metrics.auc(labels=(label > 0.5), predictions=predict)

train_step = tf.train.AdamOptimizer().minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
ss = tf.Session(config=config)
ss.run(tf.global_variables_initializer())
ss.run(tf.local_variables_initializer())
saver = tf.train.Saver(max_to_keep=10000)

training_handle = ss.run(training_iterator.string_handle())
ss.run(training_iterator.initializer)

index = 0
for loop in range(0, 10001, 1):
    index += 1
    t_epoch_start = time.time()
    for x in range(0, 100, 1):
        train_feed = {keep_prob: 0.5,
                      handle: training_handle,
                      is_train: True}
        ss.run(train_step, feed_dict=train_feed)
    feed_t = {keep_prob: 1.0,
              handle: training_handle,
              is_train: False}
    t_epoch_end = time.time()
    ckp_path = os.path.join(FLAGS.checkpointDir, "model.ckpt" + str(index))
    save_path = saver.save(ss, ckp_path)
    print(str(loop) + ' train_auc&loss:' + str(ss.run([auc, loss, loss2], feed_dict=feed_t)))
    print("training data! Time used: %.2f s" % (t_epoch_end - t_epoch_start))