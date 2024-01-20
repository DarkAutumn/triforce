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
            "features" : gym.spaces.MultiBinary(1)
        })
        self.num_enemy_vectors = num_direction_vectors

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        augmented_observation = self.augment_observation(observation, info)
        return augmented_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset()
        return self.augment_observation(observation, None), info

    def augment_observation(self, observation, info):
        # Extract features and store them in the dictionary format
        vectors = self.get_enemy_vectors(info)
        features = self.get_features(info)
        return {"image": observation, "vectors": vectors, "features": features}

    def get_enemy_vectors(self, info):
        if info is None or 'link_pos' not in info or 'objects' not in info:
            return np.zeros((self.num_enemy_vectors, 2), dtype=np.float32)
        
        objective = info['objective_vector']
        closest_enemy = info['closest_enemy_vector']
        closest_projectile = info['closest_projectile_vector']
        closest_item = info['closest_item_vector']

        # create an np array of the vectors
        normalized_vectors = [objective, closest_enemy, closest_projectile, closest_item, np.zeros(2, dtype=np.float32)]
        return np.array(normalized_vectors, dtype=np.float32)

    def get_features(self, info):
        value = 0.0
        if info is not None and 'enemy_vectors' in info and info['enemy_vectors'] and info['enemy_vectors'][0][1] > 36:
            value = 1.0

        return np.array([value], dtype=np.float32)