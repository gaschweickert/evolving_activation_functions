import sys
import tensorflow as tf
print(tf.version.VERSION)
tf.config.list_physical_devices("GPU")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation
from keras.utils.generic_utils import get_custom_objects

import math

from core_unit import CORE_UNIT

class CNN:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.custom_activation_functions = None

    def load_and_prep_data(self):
        #download mnist data and split into train and test sets
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        #reshape data to fit model
        self.X_train = self.X_train.reshape(60000,28,28,1)
        self.X_test = self.X_test.reshape(10000,28,28,1)

        #one-hot encode target column
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def set_custom_activation(self, custom_activation_functions):
        self.custom_activation_functions = custom_activation_functions

    def build_and_compile(self, custom):
        if custom:
            for i, custom_af in enumerate(self.custom_activation_functions):
                get_custom_objects().update({'custom'+ str(i): Activation(custom_af.evaluate_function)})

        #create model
        self.model = Sequential()
        #add model layers
        self.model.add(Conv2D(64, kernel_size=3, activation= 'custom0' if custom else 'relu', input_shape=(28,28,1)))
        self.model.add(Conv2D(32, kernel_size=3, activation='custom1' if custom else 'relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

        #compile model using accuracy to measure model performance
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def train_and_validate(self):
        #train the model
        #self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=3, shuffle=True)
        self.model.fit(self.X_train, self.y_train, epochs=3, shuffle=True)

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test)
