"""Qt-independent bridge between the triforce environment and the debugger GUI.

This is the ONLY debugger module that imports from triforce env/scenario code.
All other debugger modules interact through this bridge.
"""
# pylint: disable=duplicate-code

from collections import OrderedDict
import math

import torch

from triforce import TrainingScenarioDefinition, ModelDefinition
from triforce.action_space import ZeldaActionSpace
from triforce.rewards import StepRewards
from triforce.zelda_env import make_zelda_env


class StepResult:
    """A result from a single environment step."""
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            self, observation, frames, state, state_change, terminated, truncated, rewards, action_mask,
            action_mask_desc):
        self.observation = observation
        self.frames = frames
        self.state = state or state_change.state
        self.state_change = state_change
        self.terminated = terminated
        self.truncated = truncated
        self.rewards = rewards
        self.action_mask = action_mask
        self.action_mask_desc = action_mask_desc

    @property
    def completed(self):
        """Returns True if the scenario is completed and needs a reset."""
        return self.terminated or self.truncated


class ModelSelector:
    """Discovers, loads, and switches between trained models."""
    def __init__(self, env, model_path, model_definition: ModelDefinition):
        self._model_path = model_path
        self._model_definition = model_definition
        self._loaded_models = OrderedDict()
        self.action_space: ZeldaActionSpace = env
        while not isinstance(self.action_space, ZeldaActionSpace):
            self.action_space = self.action_space.env

        models = [(self._model_definition.neural_net(env.observation_space, env.action_space), name, path)
                  for name, path in self._model_definition.find_available_models(self._model_path).items()]

        for network, _, path in models:
            network.load(path)

        models.sort(key=lambda x: x[0].steps_trained)

        name = None
        for network, name, path in models:
            assert name is not None
            self._loaded_models[name] = (network, path)

        network = self._model_definition.neural_net(env.observation_space, env.action_space)
        self._loaded_models["untrained"] = (network, "untrained")

        best = self._find_best_model()
        self._loaded_models["best"] = (self._loaded_models[best][0], "best")
        self._curr_index = name if name is not None else "untrained"
        self._curr = self._loaded_models[self._curr_index]

    @property
    def model_name(self):
        """The name of the current model."""
        return self._curr_index

    @property
    def model_path(self):
        """The path to the current model."""
        return self._curr[1]

    @property
    def model(self):
        """The current model."""
        return self._curr[0]

    def select(self, name):
        """Select a model by name."""
        if name not in self._loaded_models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._loaded_models.keys())}")
        self._curr_index = name
        self._curr = self._loaded_models[name]

    def next(self):
        """Selects the next model."""
        keys = list(self._loaded_models.keys())
        self._curr_index = keys[(keys.index(self._curr_index) + 1) % len(keys)]
        self._curr = self._loaded_models[self._curr_index]

    def previous(self):
        """Selects the previous model."""
        keys = list(self._loaded_models.keys())
        self._curr_index = keys[(keys.index(self._curr_index) - 1) % len(keys)]
        self._curr = self._loaded_models[self._curr_index]

    def select_by_path(self, path):
        """Select a model by its file path.  Returns True if found."""
        for name, (network, model_path) in self._loaded_models.items():
            if model_path == path:
                self._curr_index = name
                self._curr = (network, model_path)
                return True
        return False

    def _find_best_model(self):
        best_model = "untrained"
        best_score = -math.inf
        for name, (network, _) in self._loaded_models.items():
            if not network.metrics:
                continue

            score = network.metrics.get("success-rate", None)
            if score is not None and score > best_score:
                best_score = score
                best_model = name

        return best_model

    def get_probabilities(self, obs, mask):
        """Returns the probability of every action for the model."""
        if self.model.is_multihead:
            return self._get_multihead_probabilities(obs, mask)

        logits, value = self.model.forward(obs)

        if mask is not None:
            assert mask.any(dim=-1).all(), "Mask must contain at least one valid action"
            logits = logits.clone()
            invalid_mask = ~mask
            logits[invalid_mask] = -1e9

        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        result = OrderedDict()
        result['value'] = value

        for i, prob in enumerate(probabilities.squeeze()):
            taken = self.action_space.get_action_taken(i)
            result.setdefault(taken.kind, []).append((taken.direction, prob.item()))

        return result

    def _get_multihead_probabilities(self, obs, mask):
        """Returns probabilities for a MultiHeadAgent by combining the two heads."""
        action_type_logits, direction_logits, value = self.model.forward(obs)
        num_action_types = int(self.model.action_space.nvec[0])

        if mask is not None:
            multihead_mask = self.action_space.flat_mask_to_multihead(mask.squeeze(0))
            type_mask = multihead_mask[:num_action_types].unsqueeze(0)
            dir_mask = multihead_mask[num_action_types:].unsqueeze(0)

            action_type_logits = action_type_logits.clone()
            action_type_logits[~type_mask] = -1e9
            direction_logits = direction_logits.clone()
            direction_logits[~dir_mask] = -1e9

        type_probs = torch.nn.functional.softmax(action_type_logits, dim=-1).squeeze(0)
        dir_probs = torch.nn.functional.softmax(direction_logits, dim=-1).squeeze(0)

        result = OrderedDict()
        result['value'] = value

        for type_idx in range(num_action_types):
            for dir_idx in range(4):
                flat_idx = self.action_space.multihead_to_flat(type_idx, dir_idx)
                taken = self.action_space.get_action_taken(flat_idx)
                joint_prob = (type_probs[type_idx] * dir_probs[dir_idx]).item()
                result.setdefault(taken.kind, []).append((taken.direction, joint_prob))

        return result


class EnvironmentBridge:
    """Bridge between the triforce environment and the debugger.

    Handles environment creation, stepping, restarting, and model management.
    This class is Qt-independent and can be used in headless tests.
    """
    def __init__(self, model_path, model_def, scenario_def, frame_stack):
        self.scenario_def: TrainingScenarioDefinition = scenario_def
        self.model_def: ModelDefinition = model_def

        self.env = env = make_zelda_env(self.scenario_def, model_def.action_space, render_mode='rgb_array',
                                        translation=False, frame_stack=frame_stack,
                                        multihead=getattr(model_def.neural_net, 'is_multihead', False))

        self.selector = ModelSelector(self.env, model_path, model_def)

        action_space = None
        while env:
            if isinstance(env, ZeldaActionSpace):
                action_space: ZeldaActionSpace = env
                break

            env = env.env

        if not action_space:
            raise ValueError("Could not find action space in environment.")

        self.action_space: ZeldaActionSpace = action_space
        self._observation = None
        self._action_mask = None

    def restart(self):
        """Restarts the environment episode."""
        self._action_mask = None
        obs, state = self.env.reset()

        self._observation = obs
        frames = [state.info['initial_frame']]

        self._action_mask = state.info['action_mask']
        allowed_actions = self.action_space.get_allowed_actions(state, self._action_mask)  # pylint: disable=no-member

        return StepResult(obs, frames, state, None, False, False, StepRewards(), self._action_mask, allowed_actions)

    def step(self, action=None):
        """Steps the environment. If action is None, uses the model to select one."""
        if action is None:
            action_mask = self._action_mask if len(self._action_mask.shape) > 1 else self._action_mask.unsqueeze(0)
            action = self.selector.model.get_action(self._observation, action_mask)
            action = action.squeeze(0)

        if not self.action_space.is_valid_action(action, self._action_mask):  # pylint: disable=no-member
            raise ValueError(f"Invalid action {action} for action mask {self._action_mask}")

        obs, rewards, terminated, truncated, state_change = self.env.step(action)

        self._observation = obs
        frames = state_change.frames

        self._action_mask = state_change.state.info['action_mask']
        allowed_actions = self.action_space.get_allowed_actions(state_change.state, self._action_mask)  # pylint: disable=no-member

        return StepResult(obs, frames, None, state_change, terminated, truncated, rewards, self._action_mask,
                          allowed_actions)

    def is_valid_action(self, action):
        """Returns True if the action is valid for the current state."""
        return self.action_space.is_valid_action(action, self._action_mask)  # pylint: disable=no-member

    def get_probabilities(self, obs=None, mask=None):
        """Returns model action probabilities for the current or given state."""
        obs = obs if obs is not None else self._observation
        mask = mask if mask is not None else self._action_mask
        return self.selector.get_probabilities(obs, mask)

    @property
    def model_details(self):
        """Returns a human-readable string describing the current model."""
        model = self.selector.model
        metrics = model.metrics

        progress = metrics.get("success-rate", None)
        if progress is None or progress < 0.01:
            progress = metrics.get("overworld-progress", None) or metrics.get("dungeon1-progress", None)

            if progress:
                progress = f"progress: {progress:.1f}"
        else:
            progress = f"success: {progress * 100:.1f}%"

        return f"{self.selector.model_path} ({model.steps_trained:,} timesteps {progress})"

    def close(self):
        """Closes the environment and releases resources."""
        if self.env:
            self.env.close()
        self.env = None
        self._observation = None
        self._action_mask = None
        self.selector = None

    def __del__(self):
        if self.env:
            self.close()
