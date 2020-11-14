
import os
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import layers

import datetime

from utils import *


DATASET_ROOT = '../dataset'
TRAIN_DATA = os.path.join(DATASET_ROOT, 'train')
VAL_DATA = os.path.join(DATASET_ROOT, 'validation')
EVAL_DATA = os.path.join(DATASET_ROOT, 'test')

patience = 20
imageSizeX = 256
imageSizeY = 256
batchSize = 32


def datasets():
    train = ImageDataGenerator(
        rescale=1/255. ,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
    ).flow_from_directory(
        batch_size=batchSize,
        directory=TRAIN_DATA,
        shuffle=True,
        target_size=(imageSizeX, imageSizeY), 
        subset="training",
        class_mode='binary'
    )

    validation = ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        batch_size=batchSize,
        directory=VAL_DATA,
        shuffle=True,
        target_size=(imageSizeX, imageSizeY), 
        subset="training",
        class_mode='binary'
    )
    return train, validation


def create_model():
    model = keras.Sequential([

        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(imageSizeX, imageSizeY, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation ='sigmoid')
    ])
    return model



train_data, val_data = datasets()
model = create_model()


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


cb = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=patience,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)


model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_data, 
    epochs=100, 
    validation_data=val_data,
    callbacks=[tensorboard_callback]
)

model.save('cats_dogs_auto_multiConvLayer.h5')
