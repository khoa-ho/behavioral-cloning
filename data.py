import os
import cv2
import numpy as np
from sklearn.utils import shuffle


rows, cols, chs = 14, 40, 1
SHAPE = (rows, cols, chs) # Trimmed image format

corrections = [0.0, 0.2, -0.2]

def preprocess(image):
    image_cropped = image[68:138,:,:]
    image_resized = cv2.resize(image_cropped, (cols, rows), interpolation=cv2.INTER_AREA)
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)[:,:,1] # convert to HSV and keep only the S channel
    return image_hsv.reshape(SHAPE)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples: # for each line in the driving_log
                for camera in range(3):
                    name = os.path.join('IMG', (batch_sample[camera]).split('\\')[-1]) # '\\' for windows, '/' for linux
                    image = preprocess(cv2.imread(name))
                    image_flipped = np.fliplr(image) # flip image horizontally
                    angle = float(batch_sample[3]) + corrections[camera]
                    angle_flipped = angle * (-1)
                    images.extend([image, image_flipped])
                    angles.extend([angle, angle_flipped])
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)