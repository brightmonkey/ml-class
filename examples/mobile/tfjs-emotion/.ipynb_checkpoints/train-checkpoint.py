# Import layers
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.callbacks import Callback
import pandas as pd
import numpy as np
import cv2
import keras
import subprocess
import os
import time

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

# set hyperparameters
config.batch_size = 32
config.num_epochs = 5
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dropout = 0.15
config.dense_layer_size = 128
input_shape = (48, 48, 1)


class Perf(Callback):
    """Performance callback for logging inference time"""

    def __init__(self, testX):
        self.testX = testX

    def on_epoch_end(self, epoch, logs):
        start = time.time()
        self.model.predict(self.testX)
        end = time.time()
        self.model.predict(self.testX[:1])
        latency = time.time() - end
        wandb.log({"avg_inference_time": (end - start) /
                   len(self.testX) * 1000, "latency": latency * 1000}, commit=False)


def load_fer2013():
    """Load the emotion dataset"""
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output(
            "curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz", shell=True)
    print("Loading dataset...")
    if not os.path.exists('face_cache.npz'):
        data = pd.read_csv("fer2013/fer2013.csv")
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = np.asarray(pixel_sequence.split(
                ' '), dtype=np.uint8).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), (width, height))
            faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()

        val_faces = faces[int(len(faces) * 0.8):]
        val_emotions = emotions[int(len(faces) * 0.8):]
        train_faces = faces[:int(len(faces) * 0.8)]
        train_emotions = emotions[:int(len(faces) * 0.8)]
        np.savez('face_cache.npz', train_faces=train_faces, train_emotions=train_emotions,
                 val_faces=val_faces, val_emotions=val_emotions)
    cached = np.load('face_cache.npz')

    return cached['train_faces'], cached['train_emotions'], cached['val_faces'], cached['val_emotions']


# loading dataset
train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape

# normalize data
train_faces = train_faces.astype('float32') / 255.0
val_faces = val_faces.astype('float32') / 255.0

# Define the model here, CHANGEME
model = Sequential()
model.add(keras.layers.Conv2D(32,
                                 (config.first_layer_conv_width,
                                  config.first_layer_conv_height),
                                 input_shape=(48, 48, 1),
                                 activation='relu'))
model.add(keras.layers.Conv2D(32,
                                 (config.first_layer_conv_width,
                                  config.first_layer_conv_height),
                                 activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32,
                                 (config.first_layer_conv_width,
                                  config.first_layer_conv_height),
                                 activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(keras.layers.Dense(config.dense_layer_size, activation='softmax'))

model.add(Dense(num_classes, activation="relu"))
#model.add(Dense(num_classes, activation="relu"))


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# log the number of total parameters
config.total_params = model.count_params()
model.fit(train_faces, train_emotions, batch_size=config.batch_size,
          epochs=config.num_epochs, verbose=1, callbacks=[
              Perf(val_faces),
              WandbCallback(data_type="image", labels=[
                            "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
          ], validation_data=(val_faces, val_emotions))

# save the model
model.save("emotion.h5")
