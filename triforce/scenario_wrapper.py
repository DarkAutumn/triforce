"""Wraps the environment to call our critic and end conditions."""

from collections import deque
import gzip
import os
from typing import Deque, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import gymnasium as gym
import stable_retro as retro
import torch
import yaml

from .metrics import MetricTracker
from .objectives import get_objective_selector
from .rewards import StepRewards
from .zelda_enums import Direction, MapLocation
from . import critics
from . import end_conditions

class TrainingScenarioDefinition(BaseModel):
    """A scenario in the game to train on.  This is a combination of critics and end conditions."""
    model_config = ConfigDict(populate_by_name=True)

    name : str
    scenario_selector : Optional[str] = Field(default='round-robin', alias='scenario-selector')
    objective : object = Field(default='GameCompletion', validate_default=True)
    iterations : int
    critic : str = 'GameplayCritic'
    metrics : List[str]
    end_conditions : List[str] = Field(alias='end-conditions')
    start : List[str | int]
    use_hints : Optional[bool] = Field(default=False, alias='use-hints')
    modifiers : Optional[List[str | dict]] = None
    per_reset : Optional[Dict[str, int | str]] = Field(default_factory=dict, exclude=True)
    per_step : Optional[Dict[str, int | str]] = Field(default_factory=dict, exclude=True)
    per_room : Optional[Dict[str, int | str]] = Field(default_factory=dict, exclude=True)

    @field_validator('objective', mode='before')
    @classmethod
    def objective_validator(cls, value):
        """Gets the ObjectiveSelector from name or structured dict."""
        if isinstance(value, str):
            objectives = get_objective_selector(value)
        elif isinstance(value, dict):
            kind = value.pop('kind', None)
            if kind is None:
                raise ValueError("Structured objective must have 'kind'")
            # Convert hyphenated keys to underscored kwargs
            params = {k.replace('-', '_'): v for k, v in value.items()}
            objectives = get_objective_selector(kind)
            if objectives is None:
                raise ValueError(f"Unknown objective selector {kind}")
            # Store objective class + params for deferred instantiation
            return (objectives, params)
        elif isinstance(value, tuple):
            return value
        else:
            objectives = value

        if objectives is None:
            raise ValueError(f"Unknown objective selector {value}")

        return (objectives, {})

    @field_validator('start', mode='before')
    @classmethod
    def start_validator(cls, value):
        """Gets the start location from the name."""
        all_saves = os.listdir(os.path.join(os.path.dirname(__file__), 'custom_integrations', 'Zelda-NES'))
        all_saves = [x for x in all_saves if x.endswith('.state')]
        all_saves = [os.path.splitext(x)[0] for x in all_saves]

        result = []
        for entry in value:
            if isinstance(entry, str):
                result.extend(x for x in all_saves if x.startswith(entry))
            else:
                result.extend(x for x in all_saves if x.startswith(f"{entry}_"))

        return result

    @field_validator('end_conditions', mode='before')
    @classmethod
    def end_conditions_validator(cls, value):
        """Ensures GameOver is always present in end conditions."""
        if 'GameOver' not in value:
            value = list(value) + ['GameOver']
        return value

    @field_validator('scenario_selector', mode='before')
    @classmethod
    def scenario_selector_validator(cls, value):
        """Gets the scenario selector from the name."""
        if value is None:
            return 'round-robin'

        if value in ('round-robin', 'probabilistic'):
            return value

        if value == "none":
            return 'round-robin'

        raise ValueError(f"Unknown scenario selector {value}")

    @staticmethod
    def _load_scenarios():
        """Loads the scenarios from triforce.yaml."""
        named_modifiers = TrainingCircuitDefinition._load_modifiers()  # pylint: disable=protected-access
        scenarios = {}
        script_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(script_dir, 'triforce.yaml'), encoding='utf-8') as f:
            for scenario in yaml.safe_load(f)["scenarios"]:
                scenario = TrainingScenarioDefinition(**scenario)

                # Resolve scenario-level modifiers into per_reset/per_step/per_room
                if scenario.modifiers:
                    per_reset, per_step, per_room = resolve_modifier_list(
                        scenario.modifiers, named_modifiers)
                    scenario.per_reset = per_reset
                    scenario.per_step = per_step
                    scenario.per_room = per_room

                scenarios[scenario.name] = scenario

        return scenarios

    @staticmethod
    def get(name, default=None):
        """Loads a scenario by name from triforce.yaml."""
        scenarios = TrainingScenarioDefinition._load_scenarios()
        return scenarios.get(name, default)

    @staticmethod
    def get_all():
        """Loads all scenarios from triforce.yaml."""
        return list(TrainingScenarioDefinition._load_scenarios().values())

class ModifierDefinition(BaseModel):
    """A named or inline modifier that overrides per_reset/per_step/per_room on scenarios."""
    model_config = ConfigDict(populate_by_name=True)

    per_reset : Optional[Dict[str, int | str]] = Field(default_factory=dict, alias='per-reset')
    per_step : Optional[Dict[str, int | str]] = Field(default_factory=dict, alias='per-step')
    per_room : Optional[Dict[str, int | str]] = Field(default_factory=dict, alias='per-room')


def resolve_modifier_list(modifier_list, named_modifiers):
    """Resolve a list of modifier names/inline dicts into merged per_reset/per_step/per_room.

    Each modifier is either a string (named ref) or a dict (inline definition).
    Applied in list order — later entries override earlier ones (shallow merge per key).

    Returns (per_reset, per_step, per_room) as plain dicts.
    """
    per_reset = {}
    per_step = {}
    per_room = {}

    for mod in modifier_list:
        if isinstance(mod, str):
            resolved = named_modifiers.get(mod)
            if resolved is None:
                raise ValueError(f"Unknown modifier '{mod}'")
        elif isinstance(mod, dict):
            resolved = ModifierDefinition(**mod)
        else:
            raise ValueError(f"Invalid modifier entry: {mod}")

        if resolved.per_reset:
            per_reset.update(resolved.per_reset)
        if resolved.per_step:
            per_step.update(resolved.per_step)
        if resolved.per_room:
            per_room.update(resolved.per_room)

    return per_reset, per_step, per_room

class ExitCriteria(BaseModel):
    """Exit criteria for a training circuit entry."""
    metric : str
    threshold : float

class ConditionalTrigger(BaseModel):
    """Condition that triggers a remediation scenario during weighted training."""
    metric : str
    threshold : float
    cooldown : int = 0

class TrainingCircuitEntry(BaseModel):
    """An entry in a training circuit."""
    model_config = ConfigDict(populate_by_name=True)

    scenario : Optional[str] = None
    circuit : Optional[str] = None
    iterations : Optional[int] = None
    exit_criteria : Optional[ExitCriteria] = Field(None, alias='exit-criteria')
    weight : Optional[float] = None
    condition : Optional[ConditionalTrigger] = None
    modifiers : Optional[List[str | dict]] = None

class TrainingCircuitDefinition(BaseModel):
    """A training circuit."""
    model_config = ConfigDict(populate_by_name=True)

    name : str
    kind : str = 'sequential'
    scenarios : List[TrainingCircuitEntry]
    modifiers : Optional[List[str | dict]] = None
    iterations : Optional[int] = None
    exit_criteria : Optional[ExitCriteria] = Field(None, alias='exit-criteria')

    @field_validator('kind')
    @classmethod
    def validate_kind(cls, value):
        """Validates the circuit kind."""
        if value not in ('sequential', 'weighted'):
            raise ValueError(f"Unknown circuit kind '{value}', must be 'sequential' or 'weighted'")
        return value

    @model_validator(mode='after')
    def validate_entries(self):
        """Validates weight/condition fields match circuit kind."""
        if self.kind == 'weighted':
            for entry in self.scenarios:
                name = entry.scenario or entry.circuit
                if entry.condition is not None:
                    if entry.weight is not None:
                        raise ValueError(f"Weighted circuit '{self.name}' conditional "
                                         f"entry '{name}' must not have a weight")
                elif entry.weight is None:
                    raise ValueError(f"Weighted circuit '{self.name}' entry "
                                     f"'{name}' must have a weight")
        else:
            for entry in self.scenarios:
                name = entry.scenario or entry.circuit
                if entry.weight is not None:
                    raise ValueError(f"Sequential circuit '{self.name}' entry "
                                     f"'{name}' must not have a weight")
                if entry.condition is not None:
                    raise ValueError(f"Sequential circuit '{self.name}' entry "
                                     f"'{name}' must not have a condition "
                                     f"(only weighted circuits support conditions)")
        return self

    @staticmethod
    def _load_modifiers():
        """Loads named modifier definitions from triforce.yaml."""
        script_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(script_dir, 'triforce.yaml'), encoding='utf-8') as f:
            data = yaml.safe_load(f)
        raw = data.get("modifiers", [])
        result = {}
        for item in raw:
            for name, body in item.items():
                result[name] = ModifierDefinition(**body)
        return result

    @staticmethod
    def _load_circuits():
        """Loads the training circuits from triforce.yaml."""
        named_modifiers = TrainingCircuitDefinition._load_modifiers()
        circuits = {}
        script_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(script_dir, 'triforce.yaml'), encoding='utf-8') as f:
            for circuit in yaml.safe_load(f)["training-circuits"]:
                circuit = TrainingCircuitDefinition(**circuit)

                for entry in circuit.scenarios:
                    # Each entry must have exactly one of scenario or circuit
                    if entry.scenario and entry.circuit:
                        raise ValueError(f"Circuit '{circuit.name}' entry has both "
                                         f"'scenario' and 'circuit' — use one or the other")
                    if not entry.scenario and not entry.circuit:
                        raise ValueError(f"Circuit '{circuit.name}' entry must have "
                                         f"either 'scenario' or 'circuit'")

                # Validate named modifier refs exist at both circuit and entry level
                def _validate_modifier_refs(mod_list, context):
                    if not mod_list:
                        return
                    for mod in mod_list:
                        if isinstance(mod, str) and mod not in named_modifiers:
                            raise ValueError(f"{context} references unknown modifier '{mod}'")

                _validate_modifier_refs(circuit.modifiers, f"Circuit '{circuit.name}'")
                for entry in circuit.scenarios:
                    entry_name = entry.scenario or entry.circuit
                    _validate_modifier_refs(
                        entry.modifiers,
                        f"Circuit '{circuit.name}' entry '{entry_name}'")

                circuits[circuit.name] = circuit

        # Validate no circular references
        TrainingCircuitDefinition._check_cycles(circuits)

        return circuits

    @staticmethod
    def _check_cycles(circuits):
        """Detect circular references in circuit-in-circuit nesting."""
        def visit(name, visiting):
            if name not in circuits:
                return
            if name in visiting:
                cycle = ' -> '.join(list(visiting) + [name])
                raise ValueError(f"Circular circuit reference: {cycle}")
            visiting.add(name)
            for entry in circuits[name].scenarios:
                if entry.circuit:
                    visit(entry.circuit, visiting)
            visiting.discard(name)

        for name in circuits:
            visit(name, set())

    @staticmethod
    def get(name, default=None):
        """Loads a training circuit by name from triforce.yaml."""
        circuits = TrainingCircuitDefinition._load_circuits()
        return circuits.get(name, default)

    @staticmethod
    def get_all():
        """Loads all training circuits from triforce.yaml."""
        return list(TrainingCircuitDefinition._load_circuits().values())


class WeightedScenarioSelector:
    """Centralized scenario selector for weighted circuits.

    Tracks total step counts per scenario and selects the most underweight
    scenario on each episode reset. Used from the main process directly
    (single-env) or exposed via multiprocessing.managers for worker
    subprocesses to call via RPC.
    """
    def __init__(self, scenario_names, weights):
        self._scenarios = list(scenario_names)
        self._weights = list(weights)
        self._total_weight = sum(self._weights)
        self._steps = {s: 0 for s in self._scenarios}

    def update(self, last_scenario, steps):
        """Record steps for the completed scenario and return the next scenario to run.

        Call with (None, 0) for the initial reset of each worker.
        """
        if last_scenario is not None:
            self._steps[last_scenario] += steps

        total_steps = sum(self._steps.values())
        if total_steps == 0:
            return self._scenarios[0]

        best = None
        best_deficit = -float('inf')
        for scenario, weight in zip(self._scenarios, self._weights):
            target = weight / self._total_weight
            actual = self._steps[scenario] / total_steps
            deficit = target - actual
            if deficit > best_deficit:
                best_deficit = deficit
                best = scenario

        return best

class RoomResult:
    """Tracks whether link took damage in a room."""
    def __init__(self, room, came_from, health_lost, success):
        self.room : MapLocation = room
        self.came_from : Direction = came_from
        self.health_lost = health_lost
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

        lost_health = state.link.health - self._prev_health

        def get_level_location(full_location):
            return full_location.level, full_location.value

        if ending is not None:
            if not self._skip_room:
                success = ending.startswith('success')
                result = RoomResult(state.full_location, self._direction_from, lost_health, success)
                state_change.state.info['room_result'] = result
                self._memory.append(result)
                self._prev_health = None
                self._direction_from = None

        elif get_level_location(prev.full_location) != get_level_location(state.full_location):
            success = state.full_location in prev.objectives.next_rooms
            if not self._skip_room:
                result = RoomResult(prev.full_location, self._direction_from, lost_health, success)
                state_change.state.info['room_result'] = result
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

        if not should_use_round_robin:
            # Calculate probabilities with exponential decay
            direction, location = self._select_probabilistically()
            state = self.get_name_from_direction_location(direction, location)
            full_path = os.path.join(os.path.dirname(__file__), 'custom_integrations', 'Zelda-NES', state)
            if os.path.exists(full_path):
                return state

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

    @staticmethod
    def get_name_from_direction_location(direction, location):
        """Returns the name of the state file."""
        d = direction.name[0].lower() if direction not in (None, Direction.NONE) else 'c'
        return f"{location.level}_{location.value:02x}{d}.state"

    def _select_probabilistically(self):
        decay_factor = 0.9
        weights = {}
        total_weight = 0.0

        for i, result in enumerate(reversed(self._memory)):
            weight = (decay_factor ** i) * (2.0 if not result.success else 1.0)
            if result.health_lost:
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
        return direction, location

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
    def __init__(self, env, scenario : TrainingScenarioDefinition, weighted_selector=None,
                 state_override=None):
        super().__init__(env)
        self._last_save_state = None
        self._scenario = scenario
        self._step_count = 0
        self._weighted_selector = weighted_selector
        self._state_override = state_override
        self._configure_scenario(scenario)

    def _configure_scenario(self, scenario):
        """Set up critic, end conditions, room selector, and metrics for a scenario."""
        self._scenario = scenario
        self._critic = getattr(critics, scenario.critic)()
        self._conditions = []

        # Extract objective params for end conditions that accept them
        _, obj_params = scenario.objective if isinstance(scenario.objective, tuple) else (None, {})

        for ec in scenario.end_conditions:
            ec_class = getattr(end_conditions, ec)
            try:
                self._conditions.append(ec_class(**obj_params))
            except TypeError:
                self._conditions.append(ec_class())

        # In weighted mode, pass scenario name so MetricTracker buffers per-scenario
        scenario_name = scenario.name if self._weighted_selector is not None else None
        MetricTracker.close()
        self._metrics : MetricTracker = MetricTracker(scenario.metrics, scenario_name=scenario_name)

        match scenario.scenario_selector:
            case 'round-robin':
                self.room_selector = RoundRobinSelector(scenario.start)
            case 'probabilistic':
                self.room_selector = ProbabilisticSelector(scenario.start)
            case _:
                raise ValueError(f"Unknown scenario selector {scenario.scenario_selector}")

    def switch_scenario(self, scenario : TrainingScenarioDefinition):
        """Switch to a new scenario, reconfiguring all components."""
        self._configure_scenario(scenario)
        self._find_state_change_wrapper().switch_scenario(scenario)

    def __del__(self):
        MetricTracker.close()

    def _find_state_change_wrapper(self):
        """Walk the wrapper chain to find the StateChangeWrapper."""
        from .state_change_wrapper import StateChangeWrapper as SCW  # pylint: disable=import-outside-toplevel
        env = self.env
        while env is not None:
            if isinstance(env, SCW):
                return env
            env = getattr(env, 'env', None)
        raise RuntimeError("StateChangeWrapper not found in wrapper chain")

    def reset(self, **kwargs):
        # In weighted mode, ask the selector which scenario to run next
        if self._weighted_selector is not None:
            next_scenario_name = self._weighted_selector.update(
                self._scenario.name if self._step_count > 0 else None,
                self._step_count)
            self._step_count = 0
            if next_scenario_name != self._scenario.name:
                next_scenario = TrainingScenarioDefinition.get(next_scenario_name)
                self.switch_scenario(next_scenario)

        self.room_selector.reset()

        # Use the one-shot state override if set (for debugging), otherwise ask the selector.
        if self._state_override is not None:
            save_state = self._state_override
            self._state_override = None
        else:
            save_state = self.room_selector.next()

        if save_state != self._last_save_state:
            self._last_save_state = save_state
            self.unwrapped.load_state(save_state, retro.data.Integrations.CUSTOM_ONLY)

        obs, state = super().reset(**kwargs)
        self._critic.clear()
        for ec in self._conditions:
            ec.clear()

        self._metrics.begin_scenario(state)
        return obs, state

    def step(self, action):
        self._step_count += 1

        # Step the environment
        obs, _, terminated, truncated, state_change = self.env.step(action)
        rewards = StepRewards()

        # Drop a save state if we don't have one of the current location
        if state_change.changed_location:
            self._try_save_state(state_change)

        # Critique gameplay
        self._critic.critique_gameplay(state_change, rewards)

        # Check if the scenario has ended
        end_reason = None
        for ec in self._conditions:
            ec_result = ec.is_scenario_ended(state_change)
            if ec_result is not None:
                terminated, truncated, end_reason = ec_result
                if terminated or truncated:
                    rewards.ending = end_reason
                    break

        # Update metrics
        self._metrics.step(state_change, rewards)
        if terminated or truncated:
            self._metrics.end_scenario(terminated, truncated, rewards.ending)

        # Step the room selector
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
