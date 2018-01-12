import os
import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, Lambda
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

samples = pd.read_csv('driving_log.csv', header=None).as_matrix()
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# Hyperparameters
BATCH_SIZE = 32

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = os.path.join('IMG', batch_sample[0].split('/')[0])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
valid_generator = generator(valid_samples, batch_size=BATCH_SIZE)

shape = (160, 320, 3) # Trimmed image format

# Define the model
model = Sequential()
# Normalize input
model.add(Lambda(lambda x: x/127.5 - 1.,
                 input_shape=shape,
                 output_shape=shape))
model.add(Flatten(input_shape=shape))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
                    steps_per_epoch=np.ceil(len(train_samples)/BATCH_SIZE),
                    validation_data=valid_generator,
                    validation_steps=np.ceil(len(train_samples)/BATCH_SIZE),
                    epochs=3)
