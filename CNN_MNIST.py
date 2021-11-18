import tensorflow as tf
print(tf.__version__)
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation

from keras.utils.generic_utils import get_custom_objects

import math


class CNN:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.err = 0.000001

        self.unary_units = [
        0,
        1, 
        lambda x:x, 
        lambda x:(-x), 
        lambda x:K.abs(x), 
        lambda x:x**2, 
        lambda x:x**3, 
        lambda x:K.sqrt(x), 
        lambda x:K.exp(x),  
        lambda x:K.exp(-x^2),
        lambda x:K.log(1+K.exp(x)), 
        lambda x:K.log(K.abs(x + self.err)), 
        lambda x:K.sin(x), 
        lambda x:tf.math.sinh(x), 
        lambda x:tf.math.asinh(x),
        lambda x:K.cos(x),
        lambda x:tf.math.cosh(x), 
        lambda x:tf.math.tanh(x), 
        lambda x:K.atanh(x), 
        lambda x:K.maximum(x, 0), 
        lambda x:K.minimum(x, 0),
        lambda x:(1/(1 + K.exp(-x))), 
        lambda x:tf.math.erf(x), 
        lambda x:K.sin(x)/(x+self.err)] #sinc

        self.binary_units = [
        lambda x1, x2:x1+x2, 
        lambda x1, x2:x1-x2, 
        lambda x1, x2:x1*x2, 
        lambda x1, x2:x1/(x2+ self.err), 
        lambda x1, x2:K.maximum(x1,x2), 
        lambda x1, x2:K.min(x1,x2)]

        self.custom_af= None

    def load_and_prep_data(self):
        #download mnist data and split into train and test sets
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        #reshape data to fit model
        self.X_train = self.X_train.reshape(60000,28,28,1)
        self.X_test = self.X_test.reshape(10000,28,28,1)

        #one-hot encode target column
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    @tf.function
    def custom_activation(self, x):
        if (len(self.custom_af )==3):
            bi, ui1, ui2 = self.custom_af 
            unary1 = self.unary_units[ui1](x)
            print(type(unary1))
            unary2 = self.unary_units[ui2](x)
            return self.binary_units[bi](unary1, unary2)
        else:
            print('ERROR CUSTOM ACTIVATION')
            print(self.custom_af)
            y = K.maximum(x,0)
            print(y)
            return y #relu

    def set_custom_activation(self, encoded_activation):
        self.custom_af = encoded_activation

    def build_and_compile(self):
        get_custom_objects().update({'custom1': Activation(self.custom_activation)})

        #create model
        self.model = Sequential()
        #add model layers
        self.model.add(Conv2D(64, kernel_size=3, activation='custom1', input_shape=(28,28,1), dynamic=True))
        self.model.add(Conv2D(32, kernel_size=3, activation='custom1', dynamic=True))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

        #compile model using accuracy to measure model performance
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_and_validate(self):
        #train the model
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=3)
