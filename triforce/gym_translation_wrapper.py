import gymnasium as gym

class GymTranslationWrapper(gym.Wrapper):
    """A wrapper that translates from our object-oriented observation, rewards, and actions to the environment's."""

    def reset(self, **kwargs):
        obs, state = super().reset(**kwargs)
        state.deactivate()
        return obs, state.info

    def step(self, action):
        obs, reward, terminated, truncated, change = super().step(action)
        state = change.state
        state.deactivate()
        return obs, reward.value, terminated, truncated, state.info
