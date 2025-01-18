import gymnasium as gym

from .zelda_enums import ActionKind, Direction
from .zelda_game import ZeldaGame

class TrainingHintWrapper(gym.Wrapper):
    """Training hints for early stages of training a model."""

    def reset(self, **kwargs):
        """Resets the environment."""
        obs, state = self.env.reset(**kwargs)
        self._disable_actions(None, state)
        return obs, state

    def step(self, action, **kwargs):
        """Steps the environment."""
        obs, reward, terminated, truncated, state_change = self.env.step(action, **kwargs)
        self._disable_actions(state_change.previous, state_change.state)
        return obs, reward, terminated, truncated, state_change

    def _disable_actions(self, prev : ZeldaGame, state : ZeldaGame):
        info = state.info
        if state.full_location == (0, 0x38) and state.link.tile.y < 0xa:
            info.setdefault('invalid_actions', []).append((ActionKind.MOVE, Direction.N))


        if prev is not None and prev.full_location != state.full_location:
            direction = state.full_location.get_direction_to(prev.full_location)
            info.setdefault('invalid_actions', []).append((ActionKind.MOVE, direction))
