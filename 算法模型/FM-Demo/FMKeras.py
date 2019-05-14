import keras
from keras.layers import Layer, Dense, Dropout,Input
from keras import Model,activations
from keras.optimizers import Adam
import keras.backend as K
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()["data"]
target = load_breast_cancer()["target"]

K.clear_session()
inputs = Input(shape=(30,))
out = FM(20)(inputs)
out = Dense(15,activation="sigmoid")(inputs)
out = Dense(1,activation="sigmoid")(out)

model = Model(inputs=inputs, outputs=out)
model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])
model.summary()

model.fit(data, target,
            batch_size=1,
            epochs=100,
            validation_split=0.2)