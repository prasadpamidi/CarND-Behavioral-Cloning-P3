"""
Common util methods useful for behavior cloning project.
"""

import os.path
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

def remove_zero_bias(data_frame):
    """
    Randomly deletes zero angle steering records to reduce zero angle bias.
    """
    rows_with_steering_zero = data_frame[(data_frame.steering == 0)]
    drop_indices = np.random.choice(rows_with_steering_zero.index,
                                    int(len(rows_with_steering_zero) * 0.7),
                                    replace=False)
    return data_frame.drop(drop_indices)

def process_data(directory, correction_factor):
    """
    Read the data csv file, generates image paths and measurements.
    """
    csv_file_path = directory + '/driving_log.csv'
    if not os.path.exists(csv_file_path):
        return []

    data_frame = pd.read_csv(csv_file_path,
                             names=['center', 'left', 'right',
                                    'steering', 'throttle', 'brake', 'speed'])

    data_frame = remove_zero_bias(data_frame)

    processed_results = []
    images_dir = directory + '/IMG/'
    if not os.path.exists(images_dir):
        return processed_results

    for line in data_frame.itertuples():
        if float(line.speed) < 0.1:
            continue

        steering_center = float(line.steering)
        steering_left = steering_center+correction_factor
        steering_right = steering_center-correction_factor

        img_center_path = images_dir+line.center.split('/')[-1]
        img_left_path = images_dir+line.left.split('/')[-1]
        img_right_path = images_dir+line.right.split('/')[-1]

        processed_results.append((img_center_path, steering_center))
        processed_results.append((img_left_path, steering_left))
        processed_results.append((img_right_path, steering_right))

    return processed_results

def random_brightness_image(image):
    """
    Returns an image with a random degree of brightness.
    """
    dst = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .5+np.random.uniform()
    dst[:, :, 2] = dst[:, :, 2]*random_bright
    dst = cv2.cvtColor(dst, cv2.COLOR_HSV2RGB)
    return dst

def random_shift_image(image):
    '''
    Apply warp tranform the image
    '''
    h, w, _ = image.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8, h/8)
    pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
    pts2 = np.float32([[0, horizon+v_shift], [w, horizon+v_shift], [0, h], [w, h]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(pts1, pts2),
                               (w, h), borderMode=cv2.BORDER_REPLICATE)

def generator(samples, batch_size=32, validation=False):
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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.GaussianBlur(image, (3, 3), 0)

                if not validation:
                    image = random_brightness_image(image)
                    image = random_shift_image(image)

                augmented_images.append(image)
                augmented_measurements.append(measurement)

                if abs(measurement) > 0.3:
                    image = cv2.flip(image, 1)
                    augmented_images.append(image)
                    augmented_measurements.append(measurement*-1.0)

                x_train = np.array(augmented_images)
                y_train = np.array(augmented_measurements)

                yield shuffle(x_train, y_train)
                x_train, y_train = [], []

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
