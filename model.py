import csv
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def process_data(directory, correction_factor):
    """
    Read the data csv file, generates image paths and measurements.
    """
    lines = []
    csv_file_path = directory + '/driving_log.csv'
    if not os.path.exists(csv_file_path):
        return []

    with open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    processed_results = []

    images_dir = directory + '/IMG/'
    if not os.path.exists(images_dir):
        return processed_results

    for line in lines:
        steering_center = float(line[3])
        steering_left = steering_center+correction_factor
        steering_right = steering_center-correction_factor

        img_center_path = images_dir+line[0].split('/')[-1]
        img_left_path = images_dir+line[1].split('/')[-1]
        img_right_path = images_dir+line[2].split('/')[-1]

        processed_results.append((img_center_path, steering_center))
        processed_results.append((img_left_path, steering_left))
        processed_results.append((img_right_path, steering_right))

    return processed_results

def random_brightness_image(image):
    """
    Returns an image with a random degree of brightness.
    """
    dst = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    dst[:, :, 2] = dst[:, :, 2]*random_bright
    dst = cv2.cvtColor(dst, cv2.COLOR_HSV2RGB)
    return dst

def random_translate_image(image):
    '''
    Translates the image with a random value
    '''
    rows, cols = image.shape[0], image.shape[1]

    translation_x = 10*np.random.uniform()-5
    translation_y = 10*np.random.uniform()-5
    M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    return cv2.warpAffine(image, M, (cols, rows))

def random_warp_image(image):
    '''
    Affine tranform the image
    '''
    rows, cols, _ = image.shape

    rnd_x = np.random.rand(3)-0.5
    rnd_x *= cols*0.065
    rnd_y = np.random.rand(3)-0.5
    rnd_y *= rows*0.065

    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1, x1], [y2, x1], [y1, x2]])
    pts2 = np.float32([[y1+rnd_y[0], x1+rnd_x[0]], [y2+rnd_y[1], x1+rnd_x[1]],
                       [y1+rnd_y[2], x2+rnd_x[2]]])

    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, M, (cols, rows))

def random_distort_image(image):
    """
    Returns an image with a random degree of brightness, translation and warp.
    """
    image = random_brightness_image(image)
    image = random_translate_image(image)
    return random_warp_image(image)

def generator(samples, batch_size=32, validation=False):
    """
    Returns a generator for the required images and augmented images from the given set of samples.
    """
    num_samples = len(samples)

    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            zero_measurement_count = 0
            _, unique_rotation_angle_counts = np.unique([x[1] for x in batch_samples],
                                                        return_counts=True)
            angles_count_mean = np.mean(unique_rotation_angle_counts)
            augmented_images, augmented_measurements = [], []

            for image_path, measurement in batch_samples:
                if abs(measurement) < 0.1:
                    zero_measurement_count += 1
                
                if abs(measurement) < 0.1 and zero_measurement_count > angles_count_mean:
                    continue
                else:
                    image = cv2.imread(image_path)
                    image = cv2.GaussianBlur(image, (3, 3), 0)

                #Converting to RGB format as drive.py input images are given in rgb format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if not validation:
                    image = random_distort_image(image)

                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

                x_train = np.array(augmented_images)
                y_train = np.array(augmented_measurements)

                yield shuffle(x_train, y_train)
                x_train, y_train = [], []

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
BATCH_SIZE = 128

data_samples = process_data('data', 0.25)
train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)

### Create seperate generators for training and validation data
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, validation=True)

### Create a keras model
keras_model = nvidia_arch_model()

### Load any previous saved checkpoint weights, if exists
if os.path.exists(KERAS_CHECKPOINT_FILE_PATH):
    keras_model.load_weights(KERAS_CHECKPOINT_FILE_PATH)
else:
    print("No prior model checkpoints exist")

### Compile and train the model using the generator function
keras_model.compile(loss='mse', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

history_object = keras_model.fit_generator(train_generator,
                                           steps_per_epoch=int(np.floor((len(train_samples))
                                                                        /BATCH_SIZE)*BATCH_SIZE),
                                           validation_data=validation_generator,
                                           validation_steps=int(np.floor((len(validation_samples))
                                                                         /BATCH_SIZE)*BATCH_SIZE),
                                           epochs=7,
                                           verbose=1,
                                           callbacks=keras_model_callbacks())
keras_model.summary()

keras_model.save('model.h5')

### Generate visualiaztion of the entire model
#plot_model(keras_model, to_file='model.png', show_shapes=True)

### Save the trained model

### Visualize the model loss over training and validation data - NOT NEEDED FOR EC2
#visualize_model_loss(history_object)
