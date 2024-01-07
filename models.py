# model parameters
import os
from random import randint
import tensorflow as tf
import numpy as np


image_height = 240
image_width = 256
image_channels = 3
num_outputs = 11 # todo: pull from gym_zelda_nomenu

frame_count = 3

class ZeldaModel():
    def __init__(self, name):
        self.name = name
        self.num_outputs = 4
        self.model = self._build_model()
        self.frame_count = frame_count
    
    def save(self, directory, iteration):
        self.model.save_weights(directory + '/' + f"{self.name}_{iteration:05d}.h5")

    def load(self, directory):
        # find the latest model
        latest = -1
        latest_file = None
        for file in os.listdir(directory):
            if file.startswith(self.name):
                name = file[len(self.name) + 1:]
                name = name[:name.find(".")]
                iteration = int(name)
                if iteration > latest:
                    latest = iteration
                    latest_file = file

        if latest_file is None:
            return False
        
        self.model.load_weights(directory + '/' + latest_file)
    

    def get_model_input(self, state):
        # Convert each frame in the state to grayscale and stack them
        grayscale = [self._rgb_to_grayscale(frame) for frame in state]
        input = np.stack(grayscale, axis=0)

        # Add a batch dimension at the start
        input = np.expand_dims(input, axis=0)
        return input

    def _rgb_to_grayscale(self, rgb_image):
        rgb_image_float = rgb_image.astype(np.float32)
        rgb_image_float /= 255.0
        grayscale = np.dot(rgb_image_float[...,:3], [0.2989, 0.5870, 0.1140])
        grayscale = np.expand_dims(grayscale, axis=-1)
        return grayscale
    
    def predict(self, frames, *args, **kwargs):
        model_input = self.get_model_input(frames)
        return self.model.predict(model_input, *args, **kwargs)
    
    def fit(self, frames, *args, **kwargs):
        model_input = self.get_model_input(frames)
        self.model.fit(model_input, *args, **kwargs)

    def get_random_action(self):
        return randint(0, num_outputs - 1)
    
    def _build_model(self):
        raise NotImplementedError()
    
class NoHitModel(ZeldaModel):
    def __init__(self):
        super().__init__("no_hit")
    
    def _build_model(self):
        # Image processing pathway with LSTM
        image_input = tf.keras.Input(shape=(3, image_height, image_width, 1))
        conv_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))(image_input)
        pool_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(conv_layers)
        conv2_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(pool_layers)
        flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv2_layers)
        lstm = tf.keras.layers.LSTM(128)(flatten)

        # hidden layers
        hidden1 = tf.keras.layers.Dense(128, activation='relu')(lstm)
        hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)

        # Output layer
        output = tf.keras.layers.Dense(4, activation='linear')(hidden2)

        # Create the model
        model = tf.keras.Model(inputs=image_input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        return model
    