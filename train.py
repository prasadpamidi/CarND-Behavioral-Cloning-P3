import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D, Convolution2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def process_data(directory, correction_factor):
    """
    Read the data csv file, generates image paths and measurements.
    """
    lines = []
    with open(directory + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    images_dir = directory + '/IMG/'
    
    processed_results = []
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

def generate_nvidia_model():
    """
    Returns the keras model for the popular NVIDIA architecture.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.65))
    model.add(Dense(50))
    model.add(Dropout(0.65))
    model.add(Dense(10))
    model.add(Dropout(0.65))
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

### Splitting samples and creating generators.
data_samples = process_data('data', 0.2)
train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)

### Compile and train the model using the generator function
nvidia_model = generate_nvidia_model()
nvidia_model.compile(loss='mse', optimizer='adam')

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

history_object = nvidia_model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)
nvidia_model.save('model.h5')

visualize_model_loss(history_object)