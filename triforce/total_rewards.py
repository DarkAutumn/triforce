import gymnasium as gym

from .rewards import StepRewards

class TotalRewardWrapper(gym.Wrapper):
    """A wrapper that calculates the total reward for the episode."""
    def __init__(self, env):
        super().__init__(env)
        self.total_reward = 0
        self.total_steps = 0
        self.values = {}
        self.counts = {}

    def reset(self, **kwargs):
        self.total_reward = 0
        self.total_steps = 0
        self.values.clear()
        self.counts.clear()
        return super().reset(**kwargs)

    def step(self, action):
        obs, rewards, terminated, truncated, state = super().step(action)
        self._update(rewards)

        if terminated or truncated:
            info = state.state.info
            info["total_reward"] = self.total_reward
            info["total_steps"] = self.total_steps
            info["reward_values"] = self.values
            info["reward_counts"] = self.counts

        return obs, rewards, terminated, truncated, state

    def _update(self, rewards : StepRewards):
        self.total_reward += rewards.value
        self.total_steps += 1
        for outcome in rewards:
            self.values[outcome.name] = self.values.get(outcome.name, 0) + outcome.value
            self.counts[outcome.name] = self.counts.get(outcome.name, 0) + 1
