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

    def normalize_vector(self, vector):
        """Normalize a 2D vector."""
        norm = np.linalg.norm(vector)
        if norm == 0: 
            return vector
        return vector / norm
    
    def get_enemy_vectors(self, info):
        if info is None or 'link_pos' not in info or 'objects' not in info:
            return np.zeros((self.num_enemy_vectors, 2), dtype=np.float32)

        link_pos = np.array(info['link_pos'], dtype=np.float32)
        objects = info['objects']

        closest_enemy = self.get_closest_normalized(link_pos, objects, objects.enumerate_enemy_ids())
        closest_item = self.get_closest_normalized(link_pos, objects, objects.enumerate_item_ids())
        closest_projectile = self.get_closest_normalized(link_pos, objects, objects.enumerate_projectile_ids())

        # create an np array of the vectors
        normalized_vectors = [np.zeros(2, dtype=np.float32), closest_enemy, closest_item, closest_projectile, np.zeros(2, dtype=np.float32)]
        return np.array(normalized_vectors, dtype=np.float32)

    def get_closest_normalized(self, link_pos, objects, ids):
        positions = [objects.get_position(id) for id in ids if id is not None]

        # Calculate vectors and distances to each enemy
        vectors_and_distances = [
            ((enemy_pos - link_pos), np.linalg.norm(enemy_pos - link_pos))
            for enemy_pos in positions
        ]

        epsilon = 1e-6  # Prevent division by zero

        # Sort by distance
        vectors_and_distances.sort(key=lambda x: x[1])
        normalized_vectors = [self.normalize_vector(v[0]) for v in vectors_and_distances if abs(v[1]) > epsilon]
        if normalized_vectors:
            return normalized_vectors[0]
        
        return np.zeros(2, dtype=np.float32)