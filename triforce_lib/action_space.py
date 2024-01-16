import gymnasium as gym
import numpy as np

class ZeldaAttackOnlyActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['LEFT', 'A'], ['RIGHT', 'A'], ['UP', 'A'], ['DOWN', 'A']]

        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        buttons = env.unwrapped.buttons

        self._decode_discrete_action = []
        for action in self.actions:
            arr = np.array([False] * env.action_space.n)
            
            for button in action:
                arr[buttons.index(button)] = True
            
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()
