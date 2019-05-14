import tensorflow as tf

# https://blog.csdn.net/songyunli1111/article/details/85100616

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']


# define class labels-->如分别表示：好评和差评
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# integer encode the documents
vocab_size = 50
encoded_docs = [tf.keras.preprocessing.text.one_hot(d, vocab_size) for d in docs]   #one_hot编码到[1,n],不包括0
print(encoded_docs)

# pad documents to a max length of 4 words
max_length = 5
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# define the model
input = tf.keras.Input(shape=(max_length, ))
x = tf.keras.layers.Embedding(vocab_size, 8, input_length=max_length)(input)    #这一步对应的参数量为50*8
print(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=input, outputs=x)
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())	#输出模型结构

# fit the model
model.fit(padded_docs, labels, epochs=100, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
loss_test, accuracy_test = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
# test the model
test = tf.keras.preprocessing.text.one_hot('good',50)
padded_test = tf.keras.preprocessing.sequence.pad_sequences([test], maxlen=max_length, padding='post')
print(model.predict(padded_test))
