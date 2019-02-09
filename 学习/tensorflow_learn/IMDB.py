import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)
y_train=np.asarray(train_labels).astype("float32")
y_test=np.asarray(test_labels).astype("float32")

print(len(x_train[0]))
print(x_train[0])

x_val =x_train[:10000]
partial_x_train = x_train[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
vocab_size = 10000

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu,input_shape=(vocab_size,)))
# model.add(tf.keras.layers.Embedding(vocab_size,16))
model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))

# model.add(tf.keras.layers.Embedding(vocab_size,16))
# model.add(tf.keras.layers.GlobalAveragePooling1D())
# model.add(tf.keras.layers.Dense(16,activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer="rmsprop",loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val),verbose=1)

history_dict = history.history
print(history_dict)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# plt.plot(epochs,loss,"bo",label="Training Loss")
# plt.plot(epochs,val_loss,"b",label="Validation Loss")
plt.plot(epochs,acc,"bo",label="Training acc")
plt.plot(epochs,val_acc,"b",label="Validation acc")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model.fit(x_train,  y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)