import tensorflow as tf

class FM(tf.keras.layers.Layer):
    def __init__(self, output_dim=30, activation="relu", **kwargs):
        self.output_dim = output_dim
        self.activate = activations.get(activation)
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.wight = self.add_weight(name='wight',
                                     shape=(input_shape[1], self.output_dim),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='zeros',
                                    trainable=True)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(FM, self).build(input_shape)

    def call(self, x):
        feature = K.dot(x, self.wight) + self.bias
        a = K.pow(K.dot(x, self.kernel), 2)
        b = K.dot(x, K.pow(self.kernel, 2))
        cross = K.mean(a - b, 1, keepdims=True) * 0.5
        cross = K.repeat_elements(K.reshape(cross, (-1, 1)), self.output_dim, axis=-1)
        return self.activate(feature + cross)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
