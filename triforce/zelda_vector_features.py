from typing import Sequence
import gymnasium as gym
import numpy as np

from .game_state_change import ZeldaStateChange
from .zelda_enums import Direction
from .objectives import Objective, ObjectiveKind
from .zelda_objects import ZeldaObject
from .zelda_game import ZeldaGame

OBJECT_KINDS = 3
OBJECTS_PER_KIND = 2
BOOLEAN_FEATURES = 15
DISTANCE_SCALE = 100.0

# Object features are defined as a normalized vector (x, y) and a distance scaled to [0, 1]:
#   0: closest enemy tile
#   1: closest projectile tile
#   2: closest item tile

# Objective features:
#   0: move_north
#   1: move_south
#   2: move_east
#   3: move_west
#   4: get_item
#   5: fight_enemies

# Source location:
#   0: from_north
#   1: from_south
#   2: from_east
#   3: from_west

# Boolean features:
#   0 has_beams: whether link has beams available
#   1 has_enemies: whether there are active enemies

class ZeldaVectorFeatures(gym.Wrapper):
    """A wrapper that adds additional (non-image) features to the observation space."""
    def __init__(self, env):
        super().__init__(env)
        # Original image observation space
        self.image_obs_space = env.observation_space
        # Define a new observation space as a dictionary
        self.observation_space = gym.spaces.Dict({
            "image": self.image_obs_space,
            "objects" : gym.spaces.Box(low=-1.0, high=1.0, shape=(OBJECT_KINDS, OBJECTS_PER_KIND, 3), dtype=np.float32),
            "information" : gym.spaces.MultiBinary(BOOLEAN_FEATURES)
        })

        self._prev_loc = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        state_change : ZeldaStateChange = info['state_change']

        if state_change.previous.full_location != state_change.current.full_location:
            self._prev_loc = state_change.current.full_location

        augmented_observation = self._augment_observation(observation, state_change.current)
        return augmented_observation, reward, terminated, truncated, info

    def reset(self, **_):
        observation, info = self.env.reset()
        state = info['state']
        self._prev_loc = state.full_location
        return self._augment_observation(observation, state), info

    def _augment_observation(self, observation, state):
        vectors = self._get_vectors(state)
        information = self._get_information(state)

        return {"image": observation, "vectors": vectors, "information": information}

    def _get_vectors(self, state : ZeldaGame):
        vectors = np.zeros(shape=(OBJECT_KINDS, OBJECTS_PER_KIND, 3), dtype=np.float32)

        items = state.items
        if state.treasure_location is not None:
            items = items + [ZeldaObject(state, -1, None, state.treasure_location)]

        kinds = [state.active_enemies, state.projectiles, items]
        for i, kind in enumerate(kinds):
            if kind:
                vectors[i, :OBJECTS_PER_KIND, :] = self._get_object_vectors(kind, OBJECTS_PER_KIND)

        return vectors

    def _get_object_vectors(self, objects : Sequence[ZeldaObject], count):
        result = np.zeros((count, 3), dtype=np.float32)
        result[:, 2] = -1

        # closest objects first
        objects = sorted(objects, key=lambda obj: obj.distance)
        for i, obj in enumerate(objects[:count]):
            if np.isclose(obj.distance, 0, atol=1e-5):
                result[i] = [0, 0, -1]

            else:
                dist = np.clip(obj.distance / DISTANCE_SCALE, 0, 1)
                result[i] = [obj.vector[0], obj.vector[1], dist]

        return result

    def _get_information(self, state : ZeldaGame):
        objectives = self._get_objectives_vector(state, state.objectives)

        source_direction = np.zeros(4, dtype=np.float32)
        self._assign_direction(source_direction, self._prev_loc.get_direction_of_movement(state.full_location))

        features = np.zeros(2, dtype=np.float32)
        features[0] = 1.0 if state.active_enemies else 0.0
        features[1] = 1.0 if state.link.are_beams_available else 0.0

        return np.concatenate([objectives, source_direction, features])

    def _get_objectives_vector(self, state : ZeldaGame, objectives : Objective):
        result = np.zeros(6, dtype=np.float32)

        match objectives.kind:
            case ObjectiveKind.MOVE:
                for next_room in objectives.next_rooms:
                    direction = state.full_location.get_direction_of_movement(next_room)
                    self._assign_direction(result, direction)

            case ObjectiveKind.ITEM:
                result[4] = 1.0

            case ObjectiveKind.TREASURE:
                result[4] = 1.0

            case ObjectiveKind.FIGHT:
                result[5] = 1.0

        return result

    def _assign_direction(self, result, direction):
        match direction:
            case Direction.N:
                result[0] = 1.0
            case Direction.S:
                result[1] = 1.0
            case Direction.E:
                result[2] = 1.0
            case Direction.W:
                result[3] = 1.0
