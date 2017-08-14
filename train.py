import csv
import cv2
import numpy as np
import sklearn

lines = []
with open('data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

images = []
measurements = []
correction = 0.2 # this is a parameter to tune

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for line in batch_samples:
                steering_center = float(line[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
    
                img_center = cv2.imread('data/IMG/' + line[0].split('/')[-1])
                img_left = cv2.imread('data/IMG/' + line[1].split('/')[-1])
                img_right = cv2.imread('data/IMG/' + line[2].split('/')[-1])

                images.append(img_center)
                measurements.append(steering_center)
                images.append(img_left)
                measurements.append(steering_left)
                images.append(img_right)
                measurements.append(steering_right)

            augmented_images, augmented_measurements = [], []

            for (image, measurement) in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers import Cropping2D, Convolution2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5), input_shape=(160,320,3))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24,5,5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(24,5,5, activation='relu'))

model.add(Convolution2D(6, kernel_size=(5, 5), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(16, kernel_size=(5, 5), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.65))
model.add(Dense(50))
model.add(Dropout(0.65))
model.add(Dense(10))
model.add(Dropout(0.65))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model.h5')
