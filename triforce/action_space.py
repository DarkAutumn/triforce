import gymnasium as gym
import numpy as np

from .ml_torch import SelectedAction, SelectedDirection

class ZeldaActionSpace(gym.ActionWrapper):
    """A wrapper that shrinks the action space down to what's actually used in the game."""
    def __init__(self, env, kind):
        super().__init__(env)

        # movement is always allowed
        self.actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'],
                        ['LEFT', 'A'], ['RIGHT', 'A'], ['UP', 'A'], ['DOWN', 'A'],
                        ['LEFT', 'B'], ['RIGHT', 'B'], ['UP', 'B'], ['DOWN', 'B'],
                        ['UP', 'LEFT', 'B'], ['UP', 'RIGHT', 'B'], ['DOWN', 'LEFT', 'B'], ['DOWN', 'RIGHT', 'B']
                        ]

        num_action_space = 4
        if kind != 'move-only':
            num_action_space += 4

        if kind in ('directional-item', 'diagonal-item', 'all'):
            num_action_space += 4

        if kind in ('diagonal-item', 'all'):
            num_action_space += 4

        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        self.button_count = env.action_space.n
        self.buttons = env.unwrapped.buttons

        self._decode_discrete_action = []
        for action in self.actions:
            arr = np.array([False] * self.button_count)

            for button in action:
                arr[self.buttons.index(button)] = True

            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(num_action_space)

    def action(self, action):
        if isinstance(action, list) and len(action) and isinstance(action[0], str):
            arr = np.array([False] * self.button_count)
            for button in action:
                arr[self.buttons.index(button)] = True

            return arr

        return self._decode_discrete_action[action].copy()

class MultiHeadInputWrapper(gym.Wrapper):
    """A wrapper that translates the multi-head model's action output into actual buttons for retro-environments."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.MultiDiscrete([5, 3, 3])
        self.a_button = self.unwrapped.buttons.index('A')
        self.up_button = self.unwrapped.buttons.index('UP')
        self.down_button = self.unwrapped.buttons.index('DOWN')
        self.left_button = self.unwrapped.buttons.index('LEFT')
        self.right_button = self.unwrapped.buttons.index('RIGHT')
        self.button_len = len(self.unwrapped.buttons)

    def step(self, action):
        action = self._translate_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def _assign_button_from_direction(self, action,  buttons, allow_none=False):
        action = SelectedDirection(action)
        match action:
            case SelectedDirection.N: buttons[self.up_button] = True
            case SelectedDirection.S: buttons[self.down_button] = True
            case SelectedDirection.W: buttons[self.left_button] = True
            case SelectedDirection.E: buttons[self.right_button] = True
            case _:
                if not allow_none:
                    raise ValueError(f"Invalid direction: {action}")

    def _translate_action(self, actions):
        pathfinding, danger, decision = actions.squeeze(0).tolist()
        buttons = [False] * self.button_len

        match SelectedAction(decision):
            case SelectedAction.MOVEMENT:
                self._assign_button_from_direction(pathfinding, buttons)

            case SelectedAction.ATTACK:
                self._assign_button_from_direction(danger, buttons)
                buttons[self.a_button] = True

            case SelectedAction.BEAMS:
                self._assign_button_from_direction(danger, buttons)
                buttons[self.a_button] = True

            case _: raise ValueError(f"Invalid button action: {decision}")

        return buttons
