import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    file_name = source_path.split('/')[-1]
    current_path = 'data/IMG/'+file_name
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Layer 1: Convolutional 2D. Input = 32x32x3. Output = 28x28x6.
model.add(Convolution2D(6, kernel_size=(5, 5), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2: Convolutional 2D. Output = 10x10x16.
model.add(Convolution2D(16, kernel_size=(5, 5), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten. Input = 5x5x16. Output = 400.ßß
model.add(Flatten())

# Layer 3: Fully Connected. Input = 400. Output = 120.
model.add(Dense(120, activation='relu'))

# Apply dropout
model.add(Dropout(0.65))

# Layer 4: Fully Connected. Input = 120. Output = 84.
model.add(Dense(84, activation='relu'))

# Apply dropout
model.add(Dropout(0.65))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')
