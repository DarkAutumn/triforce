
class DebugReward:
    """An action to take when a reward button is clicked."""
    def __init__(self, simulate_critique, env, action, scenario, last_info, info):
        self.env = env
        self.scenario = scenario
        self.last_info = last_info
        self.info = info
        self.action = action
        self.simulate_critique = simulate_critique

    def __call__(self):
        result = self.simulate_critique(self.env, self.action, self.scenario, self.last_info, self.info)
        reward_dict, terminated, truncated, reason = result
        print(f"{reward_dict = }")
        print(f"{terminated = }")
        print(f"{truncated = }")
        print(f"{reason = }")
