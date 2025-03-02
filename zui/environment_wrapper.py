from triforce import TrainingScenarioDefinition, ModelDefinition
from triforce.action_space import ZeldaActionSpace
from triforce.rewards import StepRewards
from triforce.zelda_env import make_zelda_env

from .model_selector import ModelSelector

class StepResult:
    """A result from a step in the environment."""
    def __init__(self, observation, frames, state, state_change, terminated, truncated, rewards, action_mask,
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
    def compelted(self):
        """Returns True if the scenario is completed and needs a reset."""
        return self.terminated or self.truncated

class EnvironmentWrapper:
    """A helper class to extract the action space from an environment."""
    def __init__(self, model_path, model_def, scenario_def, frame_stack):
        self.scenario_def : TrainingScenarioDefinition = scenario_def
        self.model_def : ModelDefinition = model_def

        self.env = env = make_zelda_env(self.scenario_def, model_def.action_space, render_mode='rgb_array',
                             translation=False, frame_stack=frame_stack)

        self.selector = ModelSelector(self.env, model_path, model_def)

        action_space = None
        while env:
            if isinstance(env, ZeldaActionSpace):
                action_space : ZeldaActionSpace = env
                break

            env = env.env

        if not action_space:
            raise ValueError("Could not find action space in environment.")

        self.action_space : ZeldaActionSpace = action_space
        self._observation = None
        self._action_mask = None

    def restart(self):
        """Restarts the model."""
        self._action_mask = None
        obs, state = self.env.reset()

        self._observation = obs
        frames = [state.info['initial_frame']]

        self._action_mask = state.info['action_mask']
        allowed_actions = self.action_space.get_allowed_actions(state, self._action_mask)

        return StepResult(obs, frames, state, None, False, False, StepRewards(), self._action_mask, allowed_actions)

    def step(self, action):
        """Steps the model."""
        if action is None:
            action_mask = self._action_mask if len(self._action_mask.shape) > 1 else self._action_mask.unsqueeze(0)
            action = self.selector.model.get_action(self._observation, action_mask)

        if not self.action_space.is_valid_action(action, self._action_mask):
            raise ValueError(f"Invalid action {action} for action mask {self._action_mask}")

        obs, rewards, terminated, truncated, state_change = self.env.step(action)

        self._observation = obs
        frames = state_change.frames

        self._action_mask = state_change.state.info['action_mask']
        allowed_actions = self.action_space.get_allowed_actions(state_change.state, self._action_mask)

        return StepResult(obs, frames, None, state_change, terminated, truncated, rewards, self._action_mask,
                          allowed_actions)

    def is_valid_action(self, action):
        """Returns True if the action is valid."""
        return self.action_space.is_valid_action(action, self._action_mask)

    @property
    def model_details(self):
        """Returns the string details of the model."""
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
        """Closes the model."""
        self.env.close()
        self.env = None
        self._observation = None
        self._action_mask = None
        self.selector = None

    def __del__(self):
        if self.env:
            self.close()
