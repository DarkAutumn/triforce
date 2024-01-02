import tensorflow as tf
import numpy as np
from .zelda_frame import ZeldaFrame

# model parameters
image_height = 240
image_width = 256
image_channels = 3
num_outputs = 7      # buttons on the controller

action_threshold = 0.7 # model must be 70% confident in a button press

xl_frame_count = 3
xl_feature_count = 4

def argb_bytes_to_np_rgb(argb_array):
    # Mesen screen size
    width, height = 256, 240
    argb_image = np.frombuffer(argb_array, dtype=np.uint8).reshape((height, width, 4))

    # Convert from ARGB to RGB (discard the Alpha channel)
    rgb_image = argb_image[:, :, 1:]

    # Normalize pixel values to be between 0 and 1
    rgb_image = rgb_image / 255.0

    return rgb_image

class ZeldaModelXL:
    def __init__(self):
        self.model = self._build_model(xl_frame_count, xl_feature_count)
        self.action_threshold = action_threshold

    def save(self, path):
        print("saved model")
        self.model.save_weights(path)

    def load(self, path):
        print("loaded model")
        self.model.load_weights(path)

    def get_random_action(self):
        return np.random.beta(0.5, 0.5, num_outputs)

    def get_model_input(self, all_frames : list[ZeldaFrame]):
        """Uses the last frame to build input for the model"""

        # Get all frames that are in the same mode as the last frame
        frames = list(self.get_frames_of_same_mode(all_frames, xl_frame_count))

        # Build the image input
        image_input = np.stack([argb_bytes_to_np_rgb(f.screen) for f in frames])

        curr = frames[-1]
        gameState = curr.game_state
        sword = 0.0
        if gameState.sword:
            sword = 1.0

        feature_input = np.array([
            gameState.hearts / gameState.heart_containers,
            gameState.location_x / 16.0,
            gameState.location_y / 16.0,
            sword,
            ])
        
        # reshape
        image_input_reshaped = np.expand_dims(image_input, axis=0)
        feature_input_reshaped = np.expand_dims(feature_input, axis=0)

        return [image_input_reshaped, feature_input_reshaped]
    
    def get_frames_of_same_mode(self, frames : list[ZeldaFrame], count):
        last = frames[-1]
        yield last
        count -= 1

        mode = last.game_state.mode
        for i in range(1, len(frames)):
            if not count:
                break

            curr = frames[-i]
            if curr.game_state.mode == mode:
                last = curr
                yield last
                count -= 1
                
        while count:
            yield last
            count -= 1
        

    def _build_model(self, frame_count, feature_count):
        # Image processing pathway with LSTM
        image_input = tf.keras.Input(shape=(frame_count, image_height, image_width, image_channels))
        conv_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))(image_input)
        pool_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(conv_layers)
        conv2_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(pool_layers)
        flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv2_layers)
        lstm = tf.keras.layers.LSTM(64)(flatten)

        # Game state pathway
        state_input = tf.keras.Input(shape=(feature_count,))
        dense_state = tf.keras.layers.Dense(64, activation='relu')(state_input)

        # Combining pathway
        combined = tf.keras.layers.concatenate([lstm, dense_state])
        combined_dense = tf.keras.layers.Dense(512, activation='relu')(combined)

        # Output layer
        output = tf.keras.layers.Dense(num_outputs, activation='sigmoid')(combined_dense)

        # Create the model
        model = tf.keras.Model(inputs=[image_input, state_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        return model