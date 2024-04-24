import gymnasium as gym
import numpy as np

NUM_DIRECTION_VECTORS = 5

class ZeldaVectorFeatures(gym.Wrapper):
    """A wrapper that adds additional (non-image) features to the observation space."""
    def __init__(self, env):
        super().__init__(env)
        # Original image observation space
        self.image_obs_space = env.observation_space
        # Define a new observation space as a dictionary
        self.observation_space = gym.spaces.Dict({
            "image": self.image_obs_space,
            "vectors" : gym.spaces.Box(low=-1.0, high=1.0, shape=(NUM_DIRECTION_VECTORS, 2), dtype=np.float32),
            "features" : gym.spaces.MultiBinary(2)
        })
        self.num_enemy_vectors = NUM_DIRECTION_VECTORS

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        augmented_observation = self._augment_observation(observation, info)
        return augmented_observation, reward, terminated, truncated, info

    def reset(self, **_):
        observation, info = self.env.reset()
        return self._augment_observation(observation, info), info

    def _augment_observation(self, observation, info):
        # Extract features and store them in the dictionary format
        vectors = self._get_enemy_vectors(info)
        features = self._get_features(info)
        return {"image": observation, "vectors": vectors, "features": features}

    def _get_enemy_vectors(self, info):
        result = [np.zeros(2, dtype=np.float32)] * NUM_DIRECTION_VECTORS
        if info is None or 'objective_vector' not in info:
            return result

        result[0] = info['objective_vector']
        result[1] = self._get_first_vector(info['active_enemies'])
        result[2] = self._get_first_vector(info['projectiles'])
        result[3] = self._get_first_vector(info['items'])
        result[4] = self._get_first_vector(info['aligned_enemies'])

        # create an np array of the vectors
        return np.array(result, dtype=np.float32)

    def _get_first_vector(self, entries):
        return entries[0].vector if entries else np.zeros(2, dtype=np.float32)

    def _get_features(self, info):
        result = np.zeros(2, dtype=np.float32)

        result[0] = 1.0 if 'active_enemies' in info and info['active_enemies'] else 0.0

        if 'beams_available' in info:
            result[1] = 1.0 if info['beams_available'] else 0.0

        return result
