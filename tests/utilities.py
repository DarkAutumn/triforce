import gymnasium as gym

class RewardRecorder:
    def __init__(self):
        self.rewards = []
        self.ends = []

    def report_reward(self, source, reward):
        self.rewards.append((source, reward))

    def report_ending(self, kind):
        self.ends.append(kind)


class CriticWrapper(gym.Wrapper):
    """Wraps the environment to actually call our critics and end conditions."""
    def __init__(self, env, critics=None, end_conditions=None):
        super().__init__(env)

        assert critics or end_conditions

        self.critics = critics or []
        self.end_conditions = end_conditions or []
        self._last = None

    def reset(self, **kwargs):
        state = super().reset(**kwargs)

        for c in self.critics:
            c.clear()

        for ec in self.end_conditions:
            ec.clear()

        self._last = None
        return state
    
    def step(self, act):
        obs, rewards, terminated, truncated, state = self.env.step(act)

        if self._last is not None:
            for c in self.critics:
                rewards += c.critique_gameplay(self._last, state)

            end = [x.is_scenario_ended(self._last, state) for x in self.end_conditions]
            terminated = terminated or any((x[0] for x in end))
            truncated = truncated or any((x[1] for x in end))

        self._last = state
        return obs, rewards, terminated, truncated, state
