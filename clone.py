import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, \
                         Lambda, Conv2D, MaxPooling2D
from keras.layers.noise import AlphaDropout
from sklearn.model_selection import train_test_split
from data import SHAPE, generator

samples = pd.read_csv('driving_log.csv').as_matrix()
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# Hyperparameters
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1.0e-3

# Data
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
valid_generator = generator(valid_samples, batch_size=BATCH_SIZE)


# Define the model
#model = load_model("./model.h5") 
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=SHAPE))
model.add(Conv2D(2, (3, 3), activation='selu', kernel_initializer='lecun_normal'))
model.add(MaxPooling2D((4,4)))
model.add(AlphaDropout(0.25))
model.add(Flatten())
model.add(Dense(1))
# Train the model
model.compile(loss='mse', optimizer=optimizers.Adam(lr=LEARNING_RATE))
model.fit_generator(train_generator, 
                    steps_per_epoch=np.ceil(len(train_samples)/BATCH_SIZE),
                    validation_data=valid_generator,
                    validation_steps=np.ceil(len(train_samples)/BATCH_SIZE),
                    epochs=EPOCHS)

# Save the model
model.save("./model.h5")
