"""Qt-independent bridge between the triforce environment and the debugger GUI.

This is the ONLY debugger module that imports from triforce env/scenario code.
All other debugger modules interact through this bridge.
"""
# pylint: disable=duplicate-code

from collections import OrderedDict
import math
import os

import torch

from triforce import TrainingScenarioDefinition, ActionSpaceDefinition, ModelKindDefinition
from triforce.action_space import ZeldaActionSpace
from triforce.models import Network
from triforce.observation_wrapper import ObservationWrapper
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


def _find_pt_files(path):
    """Find all .pt files in a path (file or directory, non-recursive)."""
    if path is None:
        return []
    if os.path.isfile(path) and path.endswith('.pt'):
        return [path]
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.pt')]
    return []


class ModelSelector:
    """Discovers, loads, and switches between trained models.

    Models are lazy-loaded: only metadata is read at discovery time.
    Full weights are loaded on first access (select, next, previous, etc.).
    """
    def __init__(self, env, model_path):
        self._model_path = model_path
        self._env = env
        self.action_space: ZeldaActionSpace = env
        while not isinstance(self.action_space, ZeldaActionSpace):
            self.action_space = self.action_space.env

        # _entries: ordered list of (label, path, metadata_dict, network_or_None)
        # network_or_None is None until the model is actually needed.
        self._entries = OrderedDict()

        # Discover .pt files — load only metadata, NOT full weights
        discovered = []
        pt_files = _find_pt_files(model_path)
        for pt_path in pt_files:
            try:
                metadata = Network.load_metadata(pt_path)
                model_kind_name = metadata.get("model_kind")
                action_space_name = metadata.get("action_space_name")
                if not model_kind_name or not action_space_name:
                    continue

                label = os.path.basename(pt_path)[:-3]
                discovered.append((label, pt_path, metadata))
            except Exception:  # pylint: disable=broad-exception-caught
                continue

        discovered.sort(key=lambda x: x[2].get("steps_trained", 0))

        self._network_class = None
        self._model_kind_name = None
        self._action_space_name = None

        for label, pt_path, metadata in discovered:
            self._entries[label] = {"path": pt_path, "metadata": metadata, "network": None}
            if self._network_class is None:
                self._model_kind_name = metadata.get("model_kind")
                self._action_space_name = metadata.get("action_space_name")
                mk = ModelKindDefinition.get(self._model_kind_name)
                self._network_class = mk.network_class

        # Add an untrained model (always eagerly created — it's cheap, no weights to load)
        if self._network_class:
            untrained = self._network_class(
                env.observation_space, env.action_space,
                model_kind=self._model_kind_name,
                action_space_name=self._action_space_name)
        else:
            mk = ModelKindDefinition.get_default()
            asd = ActionSpaceDefinition.get_default()
            self._network_class = mk.network_class
            self._model_kind_name = mk.name
            self._action_space_name = asd.name
            untrained = mk.network_class(
                env.observation_space, env.action_space,
                model_kind=mk.name, action_space_name=asd.name)

        self._entries["untrained"] = {"path": "untrained", "metadata": {}, "network": untrained}

        # Find best model from metadata alone (uses metrics inside the saved file)
        best = self._find_best_model()
        self._entries["best"] = self._entries[best]

        last_label = discovered[-1][0] if discovered else "untrained"
        self._curr_index = last_label

    def _load_network(self, entry):
        """Load full weights for an entry on demand."""
        if entry["network"] is not None:
            return entry["network"]

        pt_path = entry["path"]
        metadata = entry["metadata"]
        model_kind_name = metadata.get("model_kind", self._model_kind_name)
        action_space_name = metadata.get("action_space_name", self._action_space_name)

        model_kind = ModelKindDefinition.get(model_kind_name)
        network = model_kind.network_class(
            self._env.observation_space, self._env.action_space,
            model_kind=model_kind_name,
            action_space_name=action_space_name)
        network.load(pt_path)
        entry["network"] = network
        return network

    @property
    def model_name(self):
        """The name of the current model."""
        return self._curr_index

    @property
    def model_path(self):
        """The path to the current model."""
        return self._entries[self._curr_index]["path"]

    @property
    def model(self):
        """The current model (lazy-loaded on first access)."""
        entry = self._entries[self._curr_index]
        return self._load_network(entry)

    def select(self, name):
        """Select a model by name."""
        if name not in self._entries:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._entries.keys())}")
        self._curr_index = name

    def next(self):
        """Selects the next model."""
        keys = list(self._entries.keys())
        self._curr_index = keys[(keys.index(self._curr_index) + 1) % len(keys)]

    def previous(self):
        """Selects the previous model."""
        keys = list(self._entries.keys())
        self._curr_index = keys[(keys.index(self._curr_index) - 1) % len(keys)]

    def select_by_path(self, path):
        """Select a model by its file path.  Returns True if found."""
        for name, entry in self._entries.items():
            if entry["path"] == path:
                self._curr_index = name
                return True
        return False

    def _find_best_model(self):
        best_model = "untrained"
        best_score = -math.inf
        for name, entry in self._entries.items():
            metadata = entry.get("metadata", {})
            metrics = metadata.get("metrics", {})
            if not metrics and entry["network"] is not None:
                metrics = entry["network"].metrics

            score = metrics.get("success-rate", None)
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
            if mask.dim() < logits.dim():
                mask = mask.unsqueeze(0)
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
            # mask is already in multihead [K+4] format from get_action_mask()
            multihead_mask = mask.squeeze(0)
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
    def __init__(self, model_path, scenario_def, frame_stack,
                 action_space_name=None, model_kind_name=None):
        self.scenario_def: TrainingScenarioDefinition = scenario_def

        # Peek at .pt files to determine model kind, action space, and obs kind from metadata
        detected = self._detect_from_models(model_path)
        if not model_kind_name and detected.get("model_kind"):
            model_kind_name = detected["model_kind"]
        if not action_space_name and detected.get("action_space_name"):
            action_space_name = detected["action_space_name"]

        asd = ActionSpaceDefinition.get(action_space_name) if action_space_name \
            else ActionSpaceDefinition.get_default()
        mk = ModelKindDefinition.get(model_kind_name) if model_kind_name \
            else ModelKindDefinition.get_default()

        multihead = getattr(mk.network_class, 'is_multihead', False)

        # Infer obs_kind and frame_stack from the saved model's observation space
        obs_kind = 'viewport'
        if detected.get("obs_space"):
            from triforce.observation_wrapper import infer_obs_kind  # pylint: disable=import-outside-toplevel
            obs_kind, frame_stack = infer_obs_kind(detected["obs_space"])

        self.env = env = make_zelda_env(self.scenario_def, asd.actions, render_mode='rgb_array',
                                        translation=False, frame_stack=frame_stack,
                                        multihead=multihead, obs_kind=obs_kind)

        self.selector = ModelSelector(self.env, model_path)

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

        # Detect full_screen mode from the observation wrapper
        self._full_screen = False
        env_walk = self.env
        while env_walk:
            if isinstance(env_walk, ObservationWrapper):
                self._full_screen = env_walk.full_screen  # pylint: disable=no-member
                break
            env_walk = getattr(env_walk, 'env', None)

    @property
    def full_screen(self) -> bool:
        """Whether the environment is in full-screen (256×240) or cropped (240×224) mode."""
        return self._full_screen

    @staticmethod
    def _detect_from_models(model_path):
        """Peek at .pt files to detect model_kind, action_space_name, and obs_space."""
        pt_files = _find_pt_files(model_path)
        for pt_path in pt_files:
            try:
                meta = Network.load_metadata(pt_path)
                mk = meta.get("model_kind")
                asn = meta.get("action_space_name")
                if mk and asn:
                    return {"model_kind": mk, "action_space_name": asn,
                            "obs_space": meta.get("obs_space")}
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        return {}

    def restart(self):
        """Restarts the environment episode."""
        self._action_mask = None
        obs, state = self.env.reset()

        self._observation = obs
        frames = [state.info['initial_frame']]

        self._action_mask = state.info['action_mask']
        allowed_actions = self.action_space.get_allowed_actions(state, self._action_mask)  # pylint: disable=no-member

        return StepResult(obs, frames, state, None, False, False, StepRewards(), self._action_mask, allowed_actions)

    def load_state(self, state_bytes):
        """Load emulator state bytes and re-derive observation.

        Performs a normal reset first to initialize the wrapper chain,
        then injects the raw state bytes and steps a noop to re-derive
        all wrapper state from the new emulator RAM.
        """
        self.restart()
        self.env.unwrapped.em.set_state(state_bytes)

        # Step a noop to let all wrappers re-derive from the new RAM
        noop = torch.zeros(self.env.action_space.shape, dtype=torch.long)
        obs, _, _, _, state_change = self.env.step(noop)

        self._observation = obs
        state = state_change.state
        frames = state_change.frames

        self._action_mask = state.info['action_mask']
        allowed_actions = self.action_space.get_allowed_actions(state, self._action_mask)  # pylint: disable=no-member

        return StepResult(obs, frames, state, state_change, False, False,
                          StepRewards(), self._action_mask, allowed_actions)

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

    def get_attention_weights(self, obs=None):
        """Returns spatial attention weights if the model supports it, else None.

        Returns:
            numpy array of shape (num_heads, H', W') or None if the model has no attention.
        """
        obs = obs if obs is not None else self._observation
        model = self.selector.model
        if not hasattr(model, 'forward_with_attention'):
            return None

        with torch.no_grad():
            result = model.forward_with_attention(obs)
            attn = result[-1]  # Last element is always attention weights
            return attn.squeeze(0).cpu().numpy()  # (num_heads, H', W')

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
