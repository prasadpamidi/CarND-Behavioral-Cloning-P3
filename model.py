import csv
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def process_data(directory, correction_factor):
    """
    Read the data csv file, generates image paths and measurements.
    """
    lines = []
    if not os.path.exists(KERAS_CHECKPOINT_FILE_PATH):
        return []

    with open(directory + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    
    processed_results = []

    images_dir = directory + '/IMG/'
    if not os.path.exists(images_dir):
        return processed_results

    for line in lines:
        steering_center = float(line[3])
        steering_left = steering_center + correction_factor
        steering_right = steering_center - correction_factor

        img_center_path = images_dir + line[0].split('/')[-1]
        img_left_path = images_dir + line[1].split('/')[-1]
        img_right_path = images_dir + line[2].split('/')[-1]

        processed_results.append((img_center_path, steering_center))
        processed_results.append((img_left_path, steering_left))
        processed_results.append((img_right_path, steering_right))

    return processed_results

def generator(samples, batch_size=32):
    """
    Returns a generator for the required images and augmented images from the given set of samples.
    """
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            augmented_images, augmented_measurements = [], []

            for image_path, measurement in batch_samples:
                image = cv2.imread(image_path)
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            x_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield shuffle(x_train, y_train)

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

    model.add(Convolution2D(6, kernel_size=(5, 5), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, kernel_size=(5, 5), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.65))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.65))
    model.add(Dense(1))

    return model

def nvidia_arch_model():
    """
    Returns the keras model for the popular NVIDIA architecture.
    """
    model = Sequential()
    add_preprocessing_layers(model)

    model.add(Convolution2D(24, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def visualize_model_loss(hist_object):
    """
    Visualize the loss metrics for the keras model .
    """
    plt.plot(hist_object.history['loss'])
    plt.plot(hist_object.history['val_loss'])
    plt.title('Mean Squared Error Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')

    plt.show()

KERAS_CHECKPOINT_FILE_PATH = 'keras.weights.best.hdf5'

def keras_model_callbacks():
    """
    Returns an array of keras checkpoint callback.
    """
    return [ModelCheckpoint(KERAS_CHECKPOINT_FILE_PATH,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')]

### Splitting samples and creating generators.
data_samples = process_data('data', 0.2)
train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)

### Create seperate generators for training and validation data
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Create a keras model
keras_model = nvidia_arch_model()

# Load any previous saved checkpoint weights, if exists
if os.path.exists(KERAS_CHECKPOINT_FILE_PATH):
    keras_model.load_weights(KERAS_CHECKPOINT_FILE_PATH)
else:
    print("No prior model checkpoints exist")

### Compile and train the model using the generator function
keras_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history_object = keras_model.fit_generator(train_generator,
                                           steps_per_epoch=len(train_samples),
                                           validation_data=validation_generator,
                                           validation_steps=len(validation_samples),
                                           epochs=5,
                                           verbose=1,
                                           callbacks=keras_model_callbacks())
keras_model.summary()

### Save the trained model
keras_model.save('model.h5')

### Visualize the model loss over training and validation data - NOT NEEDED FOR EC2
#visualize_model_loss(history_object)
