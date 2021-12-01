import sys

'''
import tensorflow as tf

print(tf.version.VERSION)
tf.config.list_physical_devices("GPU")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


import math

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from sklearn.model_selection import KFold
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from core_unit import CORE_UNIT


class CNN:
    def __init__(self):
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.custom_activation_functions = None

    def load_and_prep_data(self, dataset):
        if dataset == "cifar10":
            #load cifar 10 dataset
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif dataset == "cifar100":
            #load cifar 10 dataset
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        #normalizing inputs from 0-255 to 0.0-1.0 
        x_train = x_train.astype('float32') 
        x_test = x_test.astype('float32') 
        self.x_train = x_train / 255.0 
        self.x_test = x_test / 255.0


        #one-hot encode target column
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)

        #self.inputs = np.concatenate((x_train, x_test), axis=0)
        #self.targets = np.concatenate((y_train, y_test), axis=0)

    def set_custom_activation(self, custom_activation_functions):
        self.custom_activation_functions = custom_activation_functions


    '''
    def build_and_compile(self, custom):
        if custom:
            for i, custom_af in enumerate(self.custom_activation_functions):
                get_custom_objects().update({'custom'+ str(i): Activation(custom_af.evaluate_function)})

        #create model
        self.model = Sequential()
        #add model layers
        self.model.add(Conv2D(64, kernel_size=3, activation= 'custom0' if custom else 'relu', input_shape=(28,28,1)))
        self.model.add(Conv2D(32, kernel_size=3, activation='custom0' if custom else 'relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

        #compile model using accuracy to measure model performance
        opt = optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    '''

    # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    def build_and_compile(self, mode, num_of_blocks): 
        if mode == 1:
            assert len(self.custom_activation_functions) == 1, "Warning: Invalid number of custom activations for homogenous custom"
        elif mode == 2:
            assert len(self.custom_activation_functions) == num_of_blocks * 2, "Warning: Number of custom activations does not match network number of layers!"
        elif mode == 3:
            assert len(self.custom_activation_functions) == num_of_blocks, "Warning: Number of custom activations does not match network number of blocks!"

        if not mode == 0: # custom
            for i, custom_af in enumerate(self.custom_activation_functions):
                get_custom_objects().update({'custom'+ str(i): Activation(custom_af.evaluate_function)})
        

        model = Sequential()
        layer_num = 0

        #num_block = len(self.custom_activation_functions) if per_block else 

        for block_num in range(1, num_of_blocks + 1):
            af = self.activation_setter(mode, block_num, layer_num)
            model.add(Conv2D(32 * block_num, (3, 3), activation=af, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3) if block_num == 1 else None))
            layer_num = layer_num + 1
            model.add(BatchNormalization())
            model.add(Conv2D(32 * block_num, (3, 3), activation=af, kernel_initializer='he_uniform', padding='same'))
            layer_num = layer_num + 1
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.1 + 0.1 * block_num))

        model.add(Flatten())
        model.add(Dense(32 * num_of_blocks, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1 + 0.1 * (num_of_blocks + 1)))
        model.add(Dense(10, activation='softmax'))

        # compile model
        opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model


    # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    def activation_setter(self, mode, block_num, layer_num):
        if mode == 0:
            return "relu"
        elif mode == 1:
            return "custom0"
        elif mode == 2:
            return "custom"+ str(layer_num)
        elif mode == 3:
            return "custom"+ str(block_num - 1)
        else:
            return None

    def summary(self):
        return self.model.summary()

    def train(self, train_inputs, train_targets, verbosity):
        #train the model
        #self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=3, shuffle=True)
        self.model.fit(train_inputs, train_targets, epochs=50, shuffle=True, verbose=verbosity)

    def evaluate(self, test_inputs, test_targets, verbosity):
        return self.model.evaluate(test_inputs, test_targets, verbose=verbosity)

    def k_fold_crossvalidation_evaluation(self, k, model, mode, num_of_blocks, verbosity):
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k, shuffle=True)

        # K-fold Cross Validation model evaluation
        val_acc_per_fold = []
        for train, val in kfold.split(self.x_train, self.y_train):
            model.build_and_compile(mode, num_of_blocks)
            model.train(self.x_train[train], self.y_train[train], verbosity)
            val_results = model.evaluate(self.x_train[val], self.y_train[val], verbosity)
            val_acc_per_fold.append(val_results[1])
        average_val_acc = sum(val_acc_per_fold)/len(val_acc_per_fold)
        return average_val_acc



