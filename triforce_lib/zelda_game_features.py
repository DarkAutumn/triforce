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
            "enemy_vectors": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_direction_vectors, 2), dtype=np.float32)
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
        return {"image": observation, "enemy_vectors": vectors}

    def get_enemy_vectors(self, info):
        if info is None or 'link_pos' not in info or 'objects' not in info:
            return np.zeros((self.num_enemy_vectors, 2), dtype=np.float32)
        closest_enemy = self.get_vector_of_closest(info['enemy_vectors'])
        closest_projectile = self.get_vector_of_closest(info['projectile_vectors'])
        closest_item = self.get_vector_of_closest(info['item_vectors'])

        # create an np array of the vectors
        normalized_vectors = [np.zeros(2, dtype=np.float32), closest_enemy, closest_projectile, closest_item, np.zeros(2, dtype=np.float32)]
        return np.array(normalized_vectors, dtype=np.float32)

    def get_vector_of_closest(self, vectors_and_distances):
        if vectors_and_distances:
            return vectors_and_distances[0][0]
        
        return np.zeros(2, dtype=np.float32)