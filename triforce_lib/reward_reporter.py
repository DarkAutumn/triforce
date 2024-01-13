class RewardReporter:
    def __init__(self):
        self._reward_count = {}
        self._end_count = {}

    def report_reward(self, source, reward):
        self._reward_count[source] = self._reward_count.get(source, 0) + reward

    def report_ending(self, kind):
        self._end_count[kind] = self._end_count.get(kind, 0) + 1

    def get_rewards_and_clear(self):
        summary = self._reward_count
        self._reward_count = {}
        return summary
    
    def get_endings_and_clear(self):
        summary = self._end_count
        self._end_count = {}
        return summary
    