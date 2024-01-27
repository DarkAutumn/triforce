import gymnasium as gym
import numpy as np

class ZeldaActionSpace(gym.ActionWrapper):
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

        if kind == 'directional-item' or kind == 'diagonal-item' or kind == 'all':
            num_action_space += 4
        
        if kind == 'diagonal-item' or kind == 'all':
            num_action_space += 4

        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        buttons = env.unwrapped.buttons

        self._decode_discrete_action = []
        for action in self.actions:
            arr = np.array([False] * env.action_space.n)
            
            for button in action:
                arr[buttons.index(button)] = True
            
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(num_action_space)

    def action(self, act):
        return self._decode_discrete_action[act].copy()
