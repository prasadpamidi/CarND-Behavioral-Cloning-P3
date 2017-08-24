"""
Keras model used to train for behavior cloning project using NVIDIA architecture.
"""

import os.path
import numpy as np
import utils

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2

from sklearn.model_selection import train_test_split

KERAS_CHECKPOINT_FILE_PATH = 'keras.weights.best.hdf5'
KERAS_MODEL_WEIGHTS_FILE_PATH = 'keras.weights.h5'
TRAIN_DATA_FOLDER = "/data"
BATCH_SIZE = 128

def add_preprocessing_layers(model):
    """
    Apply some common data preprocessing layers to the keras model.
    """
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

def lenet_arch_model():
    """
    Returns the keras model for the LENET architecture.
    """
    model = Sequential()
    add_preprocessing_layers(model)

    model.add(Convolution2D(6, kernel_size=(5, 5), padding="valid",
                            activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, kernel_size=(5, 5), padding="valid",
                            activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(120, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.65))
    model.add(Dense(84, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.65))
    model.add(Dense(1))

    return model

def nvidia_arch_model():
    """
    Returns the keras model for the popular NVIDIA architecture.
    """
    model = Sequential()
    add_preprocessing_layers(model)

    model.add(Convolution2D(24, 5, strides=(2, 2), kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, strides=(2, 2), kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, strides=(2, 2), kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, (3, 3), kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, (3, 3), kernel_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(50, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(1))

    return model

### Splitting samples and creating generators.
data_samples = utils.process_data(os.path.dirname(os.path.realpath(__file__))+TRAIN_DATA_FOLDER, 0.25)
train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)

### Create seperate generators for training and validation data
train_generator = utils.generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = utils.generator(validation_samples, batch_size=BATCH_SIZE, validation=True)

### Create a keras model
keras_model = nvidia_arch_model()

### Load any previous saved checkpoint weights, if exists
if os.path.exists(KERAS_CHECKPOINT_FILE_PATH):
    keras_model.load_weights(KERAS_CHECKPOINT_FILE_PATH)
else:
    print("No prior model checkpoints exist")

### Load any previous saved model weights, if exists
if os.path.exists(KERAS_MODEL_WEIGHTS_FILE_PATH):
    keras_model.load_weights(KERAS_MODEL_WEIGHTS_FILE_PATH)
else:
    print("No prior saved model weights exist")

### Compile and train the model using the generator function
keras_model.compile(loss='mse', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

history_object = keras_model.fit_generator(train_generator,
                                           steps_per_epoch=int(np.floor((len(train_samples))
                                                                        /BATCH_SIZE)*BATCH_SIZE),
                                           validation_data=validation_generator,
                                           validation_steps=int(np.floor((len(validation_samples))
                                                                         /BATCH_SIZE)*BATCH_SIZE),
                                           epochs=5,
                                           verbose=1,
                                           callbacks=[ModelCheckpoint(KERAS_CHECKPOINT_FILE_PATH,
                                                                      verbose=1,
                                                                      save_best_only=True)])

keras_model.save_weights(KERAS_MODEL_WEIGHTS_FILE_PATH)
keras_model.save('model.h5')

### Generate visualization of the entire model
#from keras.utils import plot_model
#plot_model(keras_model, to_file='model.png', show_shapes=True)

### Save the trained model

### Visualize the model loss over training and validation data - NOT NEEDED FOR EC2
#utils.visualize_model_loss(history_object)
