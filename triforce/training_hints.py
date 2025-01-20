import gymnasium as gym
from .zelda_enums import ActionKind, Direction
from .zelda_game import ZeldaGame

class TrainingHintWrapper(gym.Wrapper):
    """Training hints for early stages of training a model."""

    def reset(self, **kwargs):
        """Resets the environment."""
        obs, state = self.env.reset(**kwargs)
        self._disable_actions(state)
        return obs, state

    def step(self, action, **kwargs):
        """Steps the environment."""
        obs, reward, terminated, truncated, state_change = self.env.step(action, **kwargs)
        self._disable_actions(state_change.state)
        return obs, reward, terminated, truncated, state_change

    def _disable_actions(self, state : ZeldaGame):
        info = state.info
        link = state.link

        # Don't let link run past dungeon 1
        if state.full_location == (0, 0x38) and link.tile.y < 0xa:
            info.setdefault('invalid_actions', []).append((ActionKind.MOVE, Direction.N))

        # Don't let link use the wrong exit
        if not state.full_location.in_cave:
            if link.tile.x == 0:
                self._check_room_direction(state, info, Direction.W)
            elif link.tile.x == 0x1e:
                self._check_room_direction(state, info, Direction.E)
            elif link.tile.y == 0:
                self._check_room_direction(state, info, Direction.N)
            elif link.tile.y == 0x14:
                self._check_room_direction(state, info, Direction.S)

    def _check_room_direction(self, state, info, direction):
        next_room = state.full_location.get_location_in_direction(direction)
        if next_room not in state.objectives.next_rooms:
            info.setdefault('invalid_actions', []).append((ActionKind.MOVE, direction))
