import sys
import tensorflow as tf
print(tf.version.VERSION)

# Configuring settings for multi-gpu strategy
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
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import TensorBoard


'''
The CNN class is used to load and manipulate the datasets. Furthermore, it is used to build architectures,
train and test models. Here, the search train/validate and final train/test cycles are define.
'''
class CNN:
    def __init__(self, dataset):
        self.model = None
        self.dataset_id = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.custom_activation_functions = None

        # Set batch size for models depending on number of available gpus
        self.batch_size = 256 * number_of_gpus

        self.load_and_prep_data(dataset)

    # Loads, normalizes, and splits up the dataset into stubsets
    def load_and_prep_data(self, dataset_id):
        assert dataset_id in ("cifar10", "cifar100"), "Invalid dataset, check dataset_id"
        self.dataset_id = dataset_id
        if dataset_id == "cifar10":
            # Load cifar 10 dataset
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif dataset_id == "cifar100":
            # Load cifar 10 dataset
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        # Normalizing inputs from 0-255 to 0.0-1.0 
        x_train = x_train.astype('float32') / 255.0 
        x_test = x_test.astype('float32') / 255.0

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.1)

        self.x_test = x_test
        self.y_test = y_test

    # Builds and compiles custom CNN architectures according to the specfications given
    # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    def build_and_compile(self, mode, activation, no_blocks): 
        assert mode in (1,2,3), "Invalid mode, choose 1, 2, or 3"
        if type(activation) == str:
            assert mode == 1, "Invalid standard activation functions can only operate in mode 1"
            assert activation in ('relu', 'swish'), "Invalid activation function; try 'relu' or 'swish'"
        elif mode == 1 and not type(activation) == str:
            assert len(activation) == 1, "Warning: Invalid number of custom activations for homogenous custom"
        elif mode == 2:
            assert len(activation) == no_blocks * 2 + 1, "Warning: Number of custom activations does not match network number of layers!"
        elif mode == 3:
            assert len(activation) == no_blocks + 1, "Warning: Number of custom activations does not match network number of blocks + 1!"
        
        # Loads the custom activation functions for use in build
        if not type(activation) == str:
            for i, custom_af in enumerate(activation):
                get_custom_objects().update({'custom'+ str(i+1): Activation(custom_af.evaluate_function)})

        with mirrored_strategy.scope():
            model = Sequential()
            layer_num = 0

            # Creates VGG inspired architecture blocks with increasing number of filters and dropout rates
            for block_num in range(1, no_blocks + 1):
                layer_num = layer_num + 1
                no_filters = 2**(4+block_num)
                if block_num == 1:
                    model.add(Conv2D(no_filters, (3, 3), activation=None, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
                else:
                    model.add(Conv2D(no_filters, (3, 3), activation=None, kernel_initializer='he_uniform', padding='same'))
                af = self.get_custom_activation_function(mode, block_num, layer_num) if not type(activation) == str else activation
                model.add(Activation(af, name=af + '_B' + str(block_num) + '_L' + str(layer_num)))
                layer_num = layer_num + 1
                model.add(BatchNormalization())
                model.add(Conv2D(no_filters, (3, 3), kernel_initializer='he_uniform', padding='same'))
                if mode == 2: af = self.get_custom_activation_function(mode, block_num, layer_num) if not type(activation) == str else activation
                model.add(Activation(af, name=af + '_B' + str(block_num) + '_L' + str(layer_num)))
                model.add(BatchNormalization())
                model.add(MaxPooling2D((2, 2)))
                model.add(Dropout(0.1 + 0.1 * block_num))

            # Adds final output layers
            model.add(Flatten())
            model.add(Dense(no_filters, kernel_initializer='he_uniform'))
            af = self.get_custom_activation_function(mode, block_num + 1, layer_num + 1) if not type(activation) == str else activation
            model.add(Activation(af, name=af))
            model.add(BatchNormalization())
            model.add(Dropout(0.1 + 0.1 * (no_blocks + 1)))
            if self.dataset_id == "cifar10":
                model.add(Dense(10, activation='softmax'))
            elif self.dataset_id == "cifar100":
                model.add(Dense(100, activation='softmax'))

            # Compile model
            opt = optimizers.SGD(learning_rate=0.001*number_of_gpus, momentum=0.9)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    # Returns the custom activation for each layer according to mode and its block number
    # mode = 0 (homogenous relu), 1 (homogenous custom) 2 (heterogenous per layer), 3 (heterogenous per block)
    def get_custom_activation_function(self, mode, block_num, layer_num):
        if mode == 1:
            return "custom1"
        elif mode == 2:
            return "custom" + str(layer_num)
        elif mode == 3:
            return "custom" + str(block_num)
        else:
            return None

    # Sumarizes model (custom AFs names not included)
    def summary(self):
        return self.model.summary()

    # Saves model architecture representation to png (includes custom AF names)
    def visualize(self):
        dot_img_file = 'model_architecture.png'
        tf.keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)

    # One-hot encodes targets and formats data for use by multiple GPUs in distributed strategy
    def format_data(self, inputs, targets):
        #one-hot encode target column
        targets = to_categorical(targets)
        # Wrap data in Dataset objects.
        data = tf.data.Dataset.from_tensor_slices((inputs, targets))
        # The batch size must now be set on the Dataset objects.
        data = data.batch(self.batch_size)
        # Disable AutoShard.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        data = data.with_options(options)
        return data

    # Trains model according for given number of epochs
    def train(self, train_inputs, train_targets, no_epochs, verbosity, tensorboard_log=0):
        train_data = self.format_data(train_inputs, train_targets)
 
        # this callback will stop the training when there is NaN loss
        callback_nan = tf.keras.callbacks.TerminateOnNaN()
        # the following callback is used to track training using tensorboard
        callback_tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True)
        callbacks = [callback_nan]
        if tensorboard_log: callbacks.append(callback_tensorboard)
        
        # train the model
        history = self.model.fit(train_data, epochs=no_epochs, callbacks=callbacks, shuffle=True, verbose=verbosity)

    # Function used to validate model by testing it on the validation set (only used during searches)
    def validate(self, val_inputs, val_targets, verbosity):
        val_data = self.format_data(val_inputs, val_targets)
        return self.model.evaluate(val_data, verbose=verbosity)

    # Function that trains a model and then validates it (only used in searches)
    def search_test(self, candidate_activation, train_epochs, mode, no_blocks, verbosity):
        self.build_and_compile(mode, candidate_activation, no_blocks)
        if verbosity: print('Training:')
        self.train(self.x_train, self.y_train, train_epochs, verbosity)
        if verbosity: print('Validation:')
        val_results = self.validate(self.x_val, self.y_val, verbosity)
        return val_results

    # Used for testing the best candidates found in the searches. New models are created using the solutions, then they are
    # trained on the training and validation set for a number of epochs. To test their final performance, they are evaluated
    # on an unseen test set. This process is repeated k times. Note both max validation accuracy and final validation accruacy
    # for every run is returned.
    def final_test(self, k, mode, candidate_activation, no_blocks, no_epochs, verbosity, save_model=False, visualize=False, tensorboard_log=False):
        # Early stoppage when there is NaN loss
        callback_nan = tf.keras.callbacks.TerminateOnNaN()
        callback_tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True)
        callbacks = [callback_nan] 

        # Validation data added to train data
        x_train = np.concatenate((self.x_train, self.x_val), axis=0)
        y_train = np.concatenate((self.y_train, self.y_val), axis=0)
        
        train_data = self.format_data(x_train, y_train)
        test_data = self.format_data(self.x_test, self.y_test)

        results = []
        for run_i in range(k):
            print('Run: ' + str(run_i + 1) + '/' + str(k))
            self.build_and_compile(mode, candidate_activation, no_blocks)
            # Only save architecture and log first run
            if run_i == 0: 
                if tensorboard_log: callbacks.append(callback_tensorboard) 
                if save_model: self.model.save('architecture.h5')
                if visualize: self.visualize()
            hist = self.model.fit(train_data, validation_data=test_data, epochs=no_epochs, callbacks=callbacks, shuffle=True, verbose=verbosity)
            run_max_val_acc = max(hist.history['val_accuracy'])
            run_max_val_acc_index = hist.history['val_accuracy'].index(run_max_val_acc)
            run_final_val_acc = hist.history['val_accuracy'][-1]
            final_epoch = len(hist.history['loss'])
            results.append([run_i+1, run_max_val_acc_index, run_max_val_acc, run_final_val_acc])
        return results



