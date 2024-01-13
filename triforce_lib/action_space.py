import gymnasium as gym
import numpy as np

class ZeldaActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # currently do not allow the model to turn and act at the same time
        self.actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['A'], ['B']]

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
