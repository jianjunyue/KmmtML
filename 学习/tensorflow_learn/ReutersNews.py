import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

reuters = tf.keras.datasets.reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

one_tot_train_labels=vectorize_sequences(train_labels,46)
one_tot_test_labels=vectorize_sequences(test_labels,46)

print(len(one_tot_train_labels[0]))
print(one_tot_train_labels[0])

x_val =x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_tot_train_labels[:1000]
partial_y_train = one_tot_train_labels[1000:]



vocab_size = 10000

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(vocab_size,)))
model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(46,activation=tf.nn.softmax))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.compile(optimizer="rmsprop",loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

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

model.fit(x_train,  one_tot_train_labels, epochs=4, batch_size=512)
results = model.evaluate(x_test, one_tot_test_labels)
print(results)