from typing import Sequence
import gymnasium as gym
import numpy as np

from .zelda_enums import TileIndex
from .link import Link
from .zelda_objects import ZeldaObject
from .zelda_game import ZeldaGame

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
        game_state = info['state']
        vectors = self._get_enemy_vectors(game_state)
        features = self._get_features(game_state)

        return {"image": observation, "vectors": vectors, "features": features}

    def _get_enemy_vectors(self, state : ZeldaGame):
        result = [np.zeros(2, dtype=np.float32)] * NUM_DIRECTION_VECTORS

        result[0] = self._find_closest_non_zero(state.link, self._get_all_tiles(state.link, state.objectives.targets))
        result[1] = self._find_closest_non_zero(state.link, self._get_all_tiles(state.link, state.active_enemies))
        result[2] = self._find_closest_non_zero(state.link, self._get_all_tiles(state.link, state.projectiles))
        result[3] = self._find_closest_non_zero(state.link, self._get_all_tiles(state.link, state.items))

        # create an np array of the vectors
        return np.array(result, dtype=np.float32)

    def _get_all_tiles(self, link : Link, group : Sequence[ZeldaObject|TileIndex]):
        result = []
        for item in group:
            if isinstance(item, ZeldaObject):
                result.extend(x for x in item.self_tiles if x not in link.self_tiles)
            elif isinstance(item, TileIndex):
                result.append(item)
            else:
                raise ValueError(f"Unknown type: {type(item)}")

        return result

    def _find_closest_non_zero(self, link : Link, targets):
        if not targets:
            return np.zeros(2, dtype=np.float32)

        targets = np.array(targets, dtype=np.float32)
        link_tile = np.array(link.tile, dtype=np.float32)

        vectors = targets - link_tile
        dists = np.linalg.norm(vectors, axis=1)
        closest = np.argmin(dists)
        result = vectors[closest] / dists[closest]
        return result

    def _get_features(self, state : ZeldaGame):
        result = np.zeros(2, dtype=np.float32)

        result[0] = 1.0 if state.active_enemies else 0.0
        result[1] = 1.0 if state.link.are_beams_available else 0.0

        return result
