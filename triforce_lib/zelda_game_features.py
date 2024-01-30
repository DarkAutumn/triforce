import gymnasium as gym
import numpy as np

num_direction_vectors = 5

class ZeldaGameFeatures(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Original image observation space
        self.image_obs_space = env.observation_space
        # Define a new observation space as a dictionary
        self.observation_space = gym.spaces.Dict({
            "image": self.image_obs_space,
            "vectors" : gym.spaces.Box(low=-1.0, high=1.0, shape=(num_direction_vectors, 2), dtype=np.float32),
            "features" : gym.spaces.MultiBinary(2)
        })
        self.num_enemy_vectors = num_direction_vectors

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        augmented_observation = self.augment_observation(observation, info)
        return augmented_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset()
        return self.augment_observation(observation, info), info

    def augment_observation(self, observation, info):
        # Extract features and store them in the dictionary format
        vectors = self.get_enemy_vectors(info)
        features = self.get_features(info)
        return {"image": observation, "vectors": vectors, "features": features}

    def get_enemy_vectors(self, info):
        result = [np.zeros(2, dtype=np.float32)] * num_direction_vectors
        if info is None or 'link_pos' not in info or 'objects' not in info:
            return result
        
        result[0] = info['objective_vector']
        result[1] = info['closest_enemy_vector']
        result[2] = info['closest_projectile_vector']
        result[3] = info['closest_item_vector']

        # create an np array of the vectors
        return np.array(result, dtype=np.float32)

    def get_features(self, info):
        result = np.zeros(2, dtype=np.float32)

        result[0] = 1.0 if 'enemies_on_screen' in info and info['enemies_on_screen'] else 0.0

        if 'has_beams' in info:
            result[1] = 1.0 if info['has_beams'] else 0.0

        return result