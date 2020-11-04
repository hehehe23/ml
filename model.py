import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D



data_path = '/home/xxx/flow/data'

size = 150
batch_size = 32
seed = 23

train_set = image_dataset_from_directory(
    data_path,
    batch_size = batch_size,
    image_size = (size, size),
    label_mode = 'categorical',
    shuffle = True,
    seed = seed,
    validation_split = 0.1,
    subset = 'training',
)

valid_set = image_dataset_from_directory(
    data_path,
    batch_size = batch_size,
    image_size = (size, size),
    label_mode = 'categorical',
    shuffle = True,
    seed = seed,
    validation_split = 0.1,
    subset = 'validation',
)


epochs = 50

model = Sequential()

model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape = (size, size, 3)))
model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))

model.add(layers.MaxPooling2D())
model.add(Activation('relu'))

model.add(layers.Flatten())
model.add(Dense(64))

model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(4))
model.add(Activation('softmax'))


model.compile(optimizer='adam',
              loss = tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

hist = model.fit(train_set, epochs = epochs, validation_data = valid_set)


acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(15, 10))
plt.title('Loss')
plt.show()


model.save('/home/xxx/flow/custom_model.h5')
model.save_weights('/home/xxx/flow/custom_weights.h5')