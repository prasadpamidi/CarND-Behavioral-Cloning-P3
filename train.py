import csv
import cv2
import numpy as np
import sklearn

def processTrackData(directory, correctionFactor):
    lines = []
    
    with open(directory + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)

    images_dir = directory + '/IMG/'
    image_paths = []
    measurements = []

    for line in lines:
        steering_center = float(line[3])
        steering_left = steering_center + correctionFactor
        steering_right = steering_center - correctionFactor
    
        img_center_path = images_dir + line[0].split('/')[-1]
        img_left_path = images_dir + line[1].split('/')[-1]
        img_right_path = images_dir + line[2].split('/')[-1]

        image_paths.append(img_center_path)
        measurements.append(steering_center)
        image_paths.append(img_left_path)
        measurements.append(steering_left)
        image_paths.append(img_right_path)
        measurements.append(steering_right)

    return (image_paths, measurements)
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            augmented_images, augmented_measurements = [], []

            for image_path, measurement in batch_samples:
                image = cv2.imread(imagePath)
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers import Cropping2D, Convolution2D, MaxPooling2D

def nvidiaDrivingModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.65))
    model.add(Dense(50))
    model.add(Dropout(0.65))
    model.add(Dense(10))
    model.add(Dropout(0.65))
    model.add(Dense(1))

    return model

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
def visualizeModelLoss(history_object):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(processTrackData('data', 0.2)))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
model = nvidiaDrivingModel()
model.compile(loss='mse', optimizer='adam')

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)
model.save('model.h5')
visualizeModelLoss(history_object)