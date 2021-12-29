import sys

import tensorflow as tf
print(tf.version.VERSION)
'''
tf.config.list_physical_devices("GPU")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    number_of_gpus = len(gpus)
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        mirrored_strategy = tf.distribute.MirroredStrategy()
        #options = tf.data.Options()
        #options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

import matplotlib.pyplot as plt
#from keras_visualizer import visualizer
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical



class CNN:
    def __init__(self, dataset):
        self.model = None
        self.dataset_id = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.custom_activation_functions = None

        # set batch size for models
        # 64 for 1 gpu, 128 for 2 gpus...
        self.batch_size = 64 * number_of_gpus

        self.load_and_prep_data(dataset)

    def load_and_prep_data(self, dataset_id):
        assert dataset_id in ("cifar10", "cifar100"), "Invalid dataset, check dataset_id"
        self.dataset_id = dataset_id
        if dataset_id == "cifar10":
            #load cifar 10 dataset
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif dataset_id == "cifar100":
            #load cifar 10 dataset
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        #normalizing inputs from 0-255 to 0.0-1.0 
        x_train = x_train.astype('float32') / 255.0 
        x_test = x_test.astype('float32') / 255.0  

        self.x_train = x_train  
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        # one hot encoding happens after k-split

    # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    def build_and_compile(self, mode, activation, num_of_blocks): 
        assert mode in (1,2,3), "Invalid mode, choose 1, 2, or 3"
        if type(activation) == str:
            assert mode == 1, "Invalid standard activation functions can only operate in mode 1"
            assert activation in ('relu', 'swish'), "Invalid activation function; try 'relu' or 'swish'"
        elif mode == 1 and not type(activation) == str:
            assert len(activation) == 1, "Warning: Invalid number of custom activations for homogenous custom"
        elif mode == 2:
            assert len(activation) == num_of_blocks * 2, "Warning: Number of custom activations does not match network number of layers!"
        elif mode == 3:
            assert len(activation) == num_of_blocks, "Warning: Number of custom activations does not match network number of blocks!"
        
        if not type(activation) == str:
            for i, custom_af in enumerate(activation):
                get_custom_objects().update({'custom'+ str(i): Activation(custom_af.evaluate_function)})

        #with mirrored_strategy.scope():
        model = Sequential()
        layer_num = 0

        for block_num in range(1, num_of_blocks + 1):
            af = self.get_custom_activation_function(mode, block_num, layer_num) if not type(activation) == str else activation
            if block_num == 1:
                model.add(Conv2D(32 * block_num, (3, 3), activation=af, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
            else:
                model.add(Conv2D(32 * block_num, (3, 3), activation=af, kernel_initializer='he_uniform', padding='same'))
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
        if self.dataset_id == "cifar10":
            model.add(Dense(10, activation='softmax'))
        elif self.dataset_id == "cifar100":
            model.add(Dense(100, activation='softmax'))

        # compile model
        opt = optimizers.SGD(learning_rate=0.001*number_of_gpus, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model


    # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    def get_custom_activation_function(self, mode, block_num, layer_num):
        if mode == 1:
            return "custom0"
        elif mode == 2:
            return "custom"+ str(layer_num)
        elif mode == 3:
            return "custom"+ str(block_num - 1)
        else:
            return None

    def summary(self):
        return self.model.summary()

    def visualize(self):
        visualizer(self.model, format='png', view=True)

    def train(self, train_inputs, train_targets, num_epochs, verbosity):
        # These callback will stop the training when there is no improvement in
        # the loss or accuracy for three consecutive epochs.
        callback_loss = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        callback_acc = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.005, patience=3, mode='max')
        #one-hot encode target column
        train_targets = to_categorical(train_targets)
        #train the model
        history = self.model.fit(train_inputs, train_targets, epochs=num_epochs, batch_size=self.batch_size, callbacks=[callback_loss, callback_acc], shuffle=True, verbose=verbosity)
        if (len(history.history['loss']) < num_epochs): print('EARLY STOPPAGE AT EPOCH ' + str(len(history.history['loss'])) + '/' + str(num_epochs)) 

    def assess(self, inputs, targets, verbosity):
        #one-hot encode target column
        targets = to_categorical(targets)
        return self.model.evaluate(inputs, targets, self.batch_size, verbose=verbosity)

    def k_fold_crossvalidation(self, activation, k, train_epochs, mode, num_of_blocks, verbose):
        # Define the K-fold Cross Validator
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=None) # Should random state be none
        # K-fold Cross Validation model evaluation
        val_results_per_fold = []
        for train, val in kfold.split(self.x_train, self.y_train):
            self.build_and_compile(mode, activation, num_of_blocks)
            if verbose: print('Training:')
            self.train(self.x_train[train], self.y_train[train], train_epochs, verbose)
            if verbose: print('Validation:')
            val_results = self.assess(self.x_train[val], self.y_train[val], verbose)
            val_results_per_fold.append(val_results)
        #cnn.summary()
        average_val_results = np.mean(val_results_per_fold, axis=0) 
        return average_val_results

    def test(self, mode, activation, num_of_blocks, num_epochs, verbose):
        self.build_and_compile(mode, activation, num_of_blocks)
        self.train(self.x_train, self.y_train, num_epochs, verbose)
        return self.assess(self.x_test, self.y_test, verbose)





