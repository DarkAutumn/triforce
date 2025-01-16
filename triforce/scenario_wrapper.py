"""Wraps the environment to call our critic and end conditions."""

from collections import deque
import json
import gzip
import os
from typing import Deque, Dict, List, Optional, Union
from pydantic import BaseModel, field_validator
import gymnasium as gym
import retro
import torch

from .rewards import StepRewards
from .zelda_enums import Direction, MapLocation
from .zelda_game_data import zelda_game_data
from . import critics
from . import end_conditions

class TrainingScenarioDefinition(BaseModel):
    """A scenario in the game to train on.  This is a combination of critics and end conditions."""
    name : str
    description : str
    scenario_selector : Optional[str]
    critic : str
    reward_overrides : Optional[Dict[str, Union[int, float, None]]] = {}
    end_conditions : List[str]
    start : List[str]
    per_reset : Optional[Dict[str, int]] = {}
    per_frame : Optional[Dict[str, int]] = {}
    per_room : Optional[Dict[str, int | str]] = {}

    @field_validator('scenario_selector', mode='before')
    @classmethod
    def scenario_selector_validator(cls, value):
        """Gets the scenario selector from the name."""
        if value in ('round-robin', 'probabilistic'):
            return value

        if value == "none":
            return 'round-robin'

        raise ValueError(f"Unknown scenario selector {value}")

    @staticmethod
    def _load_scenarios():
        """Loads the models and scenarios from triforce.json."""
        scenarios = {}
        script_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(script_dir, 'triforce.json'), encoding='utf-8') as f:
            for scenario in json.load(f)["scenarios"]:
                scenario = TrainingScenarioDefinition(**scenario)
                scenarios[scenario.name] = scenario

        return scenarios

    @staticmethod
    def get(name, default=None):
        """Loads the models and scenarios from triforce.json."""
        scenarios = TrainingScenarioDefinition._load_scenarios()
        return scenarios.get(name, default)

    @staticmethod
    def get_all():
        """Loads the models and scenarios from triforce.json."""
        return list(TrainingScenarioDefinition._load_scenarios().values())

class RoomResult:
    """Tracks whether link took damage in a room."""
    def __init__(self, room, came_from, lost_health, success):
        self.room : MapLocation = room
        self.came_from : Direction = came_from
        self.lost_health = lost_health
        self.success = success

class RoomSelector:
    """Selects rooms."""
    def next(self):
        """Returns the next room."""
        raise NotImplementedError

    def step(self, state_change, ending : str):
        """Updates the selector with the new state."""

    def reset(self):
        """On env reset"""

class RoundRobinSelector(RoomSelector):
    """Selects rooms in a round-robin fashion."""
    def __init__(self, rooms):
        self.rooms = rooms
        self._curr_room = -1

    def next(self):
        """Returns the next room."""
        self._curr_room = (self._curr_room + 1) % len(self.rooms)
        return self.rooms[self._curr_room]

class ProbabilisticSelector(RoomSelector):
    """Selects rooms based on probabilities."""
    def __init__(self, rooms):
        self._starting_room = rooms[0]
        self.round_robin = RoundRobinSelector(rooms)
        self._memory : Deque[RoomResult] = deque(maxlen=128)
        self._prev_health = None
        self._direction_from = None
        self._skip_room = False
        self._room_directions = [self._get_room_direction_from_name(room) for room in rooms]

    def reset(self):
        """On env reset"""
        self._prev_health = None
        self._direction_from = None
        self._skip_room = False

    def step(self, state_change, ending : str):
        """Updates the selector with the new state."""
        prev = state_change.previous
        state = state_change.state

        if self._prev_health is None:
            self._prev_health = prev.link.health

        lost_health = state.link.health < self._prev_health
        if ending is not None:
            if not self._skip_room:
                success = ending.startswith('success')
                result = RoomResult(state.full_location, self._direction_from, lost_health, success)
                self._memory.append(result)
                self._prev_health = None
                self._direction_from = None

        elif prev.full_location != state.full_location:
            success = state.full_location in prev.objectives.next_rooms
            if not self._skip_room:
                result = RoomResult(prev.full_location, self._direction_from, lost_health, success)
                self._memory.append(result)

                self._prev_health = state.link.health if success else None
                self._direction_from = state.full_location.get_direction_to(prev.full_location)

            # if we didn't move to the right room, don't track the next room
            self._skip_room = not success

    def next(self):
        """Returns the next room."""
        should_use_round_robin = len(self._memory) < self._memory.maxlen
        if not should_use_round_robin:
            for room, direction in self._room_directions:
                if any(x.room == room and x.came_from == direction for x in self._memory):
                    continue

                should_use_round_robin = True
                break

        if should_use_round_robin:
            state = self.round_robin.next()
            if 's' in state:
                self._direction_from = Direction.S
            elif 'n' in state:
                self._direction_from = Direction.N
            elif 'e' in state:
                self._direction_from = Direction.E
            elif 'w' in state:
                self._direction_from = Direction.W
            else:
                self._direction_from = None

            return f"{state}.state"

        # Calculate probabilities with exponential decay
        direction, location = self._select_probabilistically()
        return self.get_name_from_direction_location(direction, location)

    @staticmethod
    def get_name_from_direction_location(direction, location):
        """Returns the name of the state file."""
        d = direction.name[0].lower() if direction is not None else ''
        return f"{location.level}_{location.value:02x}{d}.state"

    def _select_probabilistically(self):
        decay_factor = 0.9
        weights = {}
        total_weight = 0.0

        for i, result in enumerate(reversed(self._memory)):
            weight = (decay_factor ** i) * (2.0 if not result.success else 1.0)
            if result.lost_health:
                weight *= 1.5

            loc_dir = result.room, result.came_from
            weights[loc_dir] = weights.get(loc_dir, 0) + weight
            total_weight += weight

        probabilities = {loc: weight / total_weight for loc, weight in weights.items()}
        probability_tensor = torch.tensor(list(probabilities.values()))
        selected_index = torch.multinomial(probability_tensor, num_samples=1, replacement=True).item()

        locations = list(probabilities.keys())
        location, direction = locations[selected_index]
        self._direction_from = direction
        return direction,location

    def _get_room_direction_from_name(self, state):
        state = os.path.splitext(state)[0]
        match state[-1]:
            case 'n':
                direction = Direction.N
            case 's':
                direction = Direction.S
            case 'e':
                direction = Direction.E
            case 'w':
                direction = Direction.W
            case _:
                direction = None

        state = state[:-1]
        level, value = state.split('_')
        return MapLocation(int(level), int(value, 16), False), direction

class ScenarioWrapper(gym.Wrapper):
    """Wraps the environment to call our critic and end conditions."""
    def __init__(self, env, scenario : TrainingScenarioDefinition):
        super().__init__(env)

        self._scenario = scenario
        self._critic = getattr(critics, scenario.critic)()
        for k, v in scenario.reward_overrides.items():
            assert hasattr(self._critic, k)
            setattr(self._critic, k, v)

        self._conditions = [getattr(end_conditions, ec)() for ec in scenario.end_conditions]

        match scenario.scenario_selector:
            case 'round_robin':
                self.room_selector = RoundRobinSelector(scenario.start)
            case 'probabilistic':
                self.room_selector = ProbabilisticSelector(scenario.start)
            case _:
                raise ValueError(f"Unknown scenario selector {scenario.scenario_selector}")

        self.game_data = zelda_game_data
        self._last = None

    def reset(self, **kwargs):
        self.room_selector.reset()

        save_state = self.room_selector.next()
        if save_state != self._last:
            self._last = save_state
            self.unwrapped.load_state(save_state, retro.data.Integrations.CUSTOM_ONLY)

        obs, state = super().reset(**kwargs)

        self._critic.clear()
        for ec in self._conditions:
            ec.clear()

        return obs, state

    def step(self, action):
        obs, _, terminated, truncated, state_change = self.env.step(action)
        rewards = StepRewards()

        if state_change.changed_location:
            self._try_save_state(state_change)

        state = state_change.state

        self._critic.critique_gameplay(state_change, rewards)
        state.info['rewards'] = rewards

        end = (x.is_scenario_ended(state_change) for x in self._conditions)
        end = [x for x in end if x is not None]
        terminated = terminated or any((x[0] for x in end))
        truncated = truncated or any((x[1] for x in end))
        reason = [x[2] for x in end if x[2]]

        if reason:
            # I guess we could have more than one reason, but I'm not going to cover that corner case
            rewards.ending = reason[0]

        self.room_selector.step(state_change, rewards.ending)
        return obs, rewards, terminated, truncated, state_change

    def _try_save_state(self, state_change):
        state = state_change.state
        if not state.in_cave and state_change.changed_location:
            direction = state.full_location.get_direction_to(state_change.previous.full_location)
            location = state.full_location
            filename = ProbabilisticSelector.get_name_from_direction_location(direction, location)

            state = self.env.unwrapped.em.get_state()
            # this files's directory:
            full_path = os.path.join(os.path.dirname(__file__), 'custom_integrations', 'Zelda-NES', filename)
            if not os.path.exists(full_path):
                with gzip.open(full_path, 'wb') as f:
                    f.write(state)

__all__ = [ScenarioWrapper.__name__]
