import gymnasium as gym
import numpy as np

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
