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
from sklearn.model_selection import KFold
import numpy as np

import math

from core_unit import CORE_UNIT

class CNN:
    def __init__(self):
        self.model = None
        self.inputs = None
        self.targets = None

        self.custom_activation_functions = None

    def load_and_prep_data(self):
        #download mnist data and split into train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        #reshape data to fit model
        x_train = x_train.reshape(60000,28,28,1)
        x_test = x_test.reshape(10000,28,28,1)

        #one-hot encode target column
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        self.inputs = np.concatenate((x_train, x_test), axis=0)
        self.targets = np.concatenate((y_train, y_test), axis=0)

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

    def train_and_validate(self, train_inputs, train_targets, verbosity):
        #train the model
        #self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=3, shuffle=True)
        self.model.fit(train_inputs, train_targets, epochs=3, shuffle=True, verbose=verbosity)

    def evaluate(self, test_inputs, test_targets, verbosity):
        return self.model.evaluate(test_inputs, test_targets, verbose=verbosity)

    def k_fold_crossvalidation_evaluation(self, k, model, custom, verbosity):
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k, shuffle=True)

        # K-fold Cross Validation model evaluation
        acc_per_fold = []
        for train, test in kfold.split(self.inputs, self.targets):
            model.build_and_compile(custom)
            model.train_and_validate(self.inputs[train], self.targets[train], verbosity)
            results = model.evaluate(self.inputs[test], self.targets[test], verbosity)
            acc_per_fold.append(results[1])
        average_acc = sum(acc_per_fold)/len(acc_per_fold)
        return average_acc


