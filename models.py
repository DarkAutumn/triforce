# model parameters
from random import randint
import tensorflow as tf
import numpy as np


image_height = 240
image_width = 256
image_channels = 3
num_outputs = 11 # todo: pull from gym_zelda_nomenu

frame_count = 3

class ZeldaBaseModel:
    def __init__(self):
        self.model = self._build_model(frame_count)
        self.frame_count = frame_count

    def save(self, path):
        print("saved model")
        self.model.save_weights(path)

    def load(self, path):
        print("loaded model")
        self.model.load_weights(path)

    def get_random_action(self):
        return randint(0, num_outputs - 1)
    
    def predict(self, frames, *args, **kwargs):
        model_input = self.get_model_input(frames)
        return self.model.predict(model_input, *args, **kwargs)
    
    def fit(self, frames, *args, **kwargs):
        model_input = self.get_model_input(frames)
        self.model.fit(model_input, *args, **kwargs)

    def get_model_input(self, frames):
        """Uses the last frame to build input for the model"""

        # If we have less than the required number of frames, use the first frame for the missing ones
        frames = list(frames)
        if len(frames) < frame_count:
            first_frame = frames[0]
            for _ in range(frame_count - len(frames)):
                frames.insert(0, first_frame)
        
        
        image_input = np.stack(frames)

        # reshape and normalize
        image_input_reshaped = np.expand_dims(image_input, axis=0)
        normalized = image_input_reshaped / 255.0
        return normalized
    
    def _build_model(self, frame_count):
        # Image processing pathway with LSTM
        image_input = tf.keras.Input(shape=(frame_count, image_height, image_width, image_channels))
        conv_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))(image_input)
        pool_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(conv_layers)
        conv2_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(pool_layers)
        flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv2_layers)
        lstm = tf.keras.layers.LSTM(64)(flatten)

        # hidden layers
        hidden1 = tf.keras.layers.Dense(128, activation='relu')(lstm)
        hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)

        # Output layer
        output = tf.keras.layers.Dense(num_outputs, activation='linear')(hidden2)

        # Create the model
        model = tf.keras.Model(inputs=image_input, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        
        return model