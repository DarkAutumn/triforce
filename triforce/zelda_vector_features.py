import gymnasium as gym
import numpy as np

NUM_DIRECTION_VECTORS = 5
NUM_FEATURE_BINARIES = 5

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
            "features" : gym.spaces.MultiBinary(NUM_FEATURE_BINARIES)
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
        objective_vector = info['objective'].vector
        result[0] = objective_vector if objective_vector is not None else np.zeros(2, dtype=np.float32)
        result[1] = self._get_first_vector(info['active_enemies'])
        result[2] = self._get_first_vector(info['projectiles'])
        result[3] = self._get_first_vector(info['items'])
        result[4] = self._get_first_vector(info['aligned_enemies'])

        # create an np array of the vectors
        return np.array(result, dtype=np.float32)

    def _get_first_vector(self, entries):
        return entries[0].vector if entries else np.zeros(2, dtype=np.float32)

    def _get_features(self, info):
        result = np.zeros(NUM_FEATURE_BINARIES, dtype=np.float32)

        if 'beams_available' in info and info['beams_available']:
            result[0] = 1.0

        for i in range(4):
            if info['triforce'] & (1 << i):
                result[1] = 1.0

        return result
