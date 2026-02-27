
from collections import Counter
from enum import Enum
from math import ceil
from typing import Any, Iterable, Tuple

from .rewards import Reward, StepRewards
from .state_change_wrapper import StateChange

class Metric:
    """Tracks a single metric."""
    def clear(self):
        """Clears the metric."""
        raise NotImplementedError

    def enumerate_values(self) -> Iterable[Tuple[str, Any]]:
        """Returns the values of the metric."""
        raise NotImplementedError

    def begin_scenario(self, state):
        """Called when a new scenario begins."""

    def end_scenario(self, terminated, truncated, reason):
        """Called when a scenario ends."""

    def step(self, state_change : StateChange, rewards : StepRewards):
        """Called on each step of the scenario."""

class EnumMetric(Metric):
    """Tracks a metric that can take on a fixed set of values."""
    def __init__(self, base_name, percentage, is_enum = True):
        self.percentage = percentage
        self.base_name = base_name if base_name.endswith('/') else base_name + '/'
        self.values = []
        self._is_enum = is_enum

    def add(self, value):
        """Adds a value to the metric."""
        self.values.append(value)

    def clear(self):
        self.values.clear()

    def enumerate_values(self):
        total = len(self.values) if self.percentage else 1
        result  = []

        if total:
            counter = Counter(self.values)
            for key, value in counter.items():
                if self._is_enum:
                    key = key.value

                result.append((key, self.base_name + key, value))

            # preserve enum sort order
            result.sort(key = lambda x: x[0])
            for key, name, value in result:
                yield name, value / total

class AveragedMetric(Metric):
    """Tracks a metric that is averaged over time."""
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.values = []

    def add(self, value):
        """Adds a value to the running average."""
        self.values.append(value)

    def clear(self):
        return self.values.clear()

    def enumerate_values(self):
        if self.values:
            yield self.name, sum(self.values) / len(self.values)

class RoomProgressMetric(AveragedMetric):
    """Tracks the progress of the agent in discovering new rooms."""
    def __init__(self, room_map):
        super().__init__("room-progress")
        self._current = None
        self._room_map = room_map
        self._max_progress = max(room_map.values()) if room_map else 0

    @property
    def max_progress(self):
        """The maximum progress value in the room map."""
        return self._max_progress

    @property
    def episode_values(self):
        """Per-episode progress values."""
        return self.values

    def begin_scenario(self, state):
        self._current = 0

    def step(self, state_change : StateChange, rewards : StepRewards):
        state = state_change.state
        value = self._room_map.get((state.level, state.location), None)
        if value is not None and value > self._current:
            self._current = value

    def end_scenario(self, terminated, truncated, reason):
        self.add(self._current)

    def enumerate_values(self):
        yield from super().enumerate_values()
        if not self.values:
            return

        sorted_vals = sorted(self.values)
        n = len(sorted_vals)
        yield "progress/median", sorted_vals[n // 2]
        yield "progress/p25", sorted_vals[max(0, ceil(n * 0.25) - 1)]
        yield "progress/p75", sorted_vals[max(0, ceil(n * 0.75) - 1)]
        yield "progress/p90", sorted_vals[max(0, ceil(n * 0.90) - 1)]
        yield "progress/max", sorted_vals[-1]
        yield "progress/success", sum(1 for v in self.values if v >= self._max_progress) / n

class RoomResult(Enum):
    """The result of a room."""
    CORRECT_EXIT = "correct-exit"
    INCORRECT_EXIT = "incorrect-exit"
    DIED = "died"

    def __lt__(self, other):
        # Just make correct-exit the smallest
        return self.value[0] < other.value[0]

class RoomResultMetric(EnumMetric):
    """Tracks the result of the agent in a room."""
    def __init__(self):
        super().__init__("room-result", True)

    def step(self, state_change : StateChange, rewards : StepRewards):
        if state_change.state.game_over:
            self.add(RoomResult.DIED)
        elif state_change.previous.full_location != state_change.state.full_location:
            if state_change.state.full_location in state_change.previous.objectives.next_rooms:
                self.add(RoomResult.CORRECT_EXIT)
            else:
                self.add(RoomResult.INCORRECT_EXIT)

class RoomHealthChangeMetric(Metric):
    """Tracks the change in health in a room."""
    def __init__(self):
        self.gained = AveragedMetric("room-health-gained")
        self.lost = AveragedMetric("room-health-lost")

        self.curr_gained = None
        self.curr_lost = None

    def clear(self):
        self.gained.clear()
        self.lost.clear()

    def begin_scenario(self, state):
        self.curr_gained = 0
        self.curr_lost = 0

    def step(self, state_change : StateChange, rewards : StepRewards):
        prev, curr = state_change.previous, state_change.state
        if prev.full_location == curr.location or curr.game_over:
            self.curr_gained += state_change.health_gained
            self.curr_lost += state_change.health_lost

        if prev.full_location != curr.location or curr.game_over:
            self.gained.add(self.curr_gained)
            self.lost.add(self.curr_lost)

            self.curr_gained = 0
            self.curr_lost = 0

    def enumerate_values(self):
        yield from self.lost.enumerate_values()
        yield from self.gained.enumerate_values()

class SuccessMetric(AveragedMetric):
    """Tracks the success rate of the agent."""
    def __init__(self):
        super().__init__("success-rate")
        self._success = None

    def end_scenario(self, terminated, truncated, reason):
        self.add(1 if reason.startswith("success") else 0)

class RewardAverageMetric(AveragedMetric):
    """Tracks the reward of the agent."""
    def __init__(self):
        super().__init__("rewards")
        self._total = None

    def begin_scenario(self, state):
        self._total = 0

    def step(self, state_change : StateChange, rewards : StepRewards):
        self._total += rewards.value

    def end_scenario(self, terminated, truncated, reason):
        self.add(self._total)

class RewardDetailsMetric(Metric):
    """Tracks the details of the reward."""
    def __init__(self):
        self._rewards = {}
        self._punishments = {}

        self._curr_rewards = {}
        self._curr_punishments = {}

    def clear(self):
        for value in self._rewards.values():
            value.clear()

        for value in self._punishments.values():
            value.clear()

    def begin_scenario(self, state):
        self._curr_rewards.clear()
        self._curr_punishments.clear()

    def step(self, state_change : StateChange, rewards : StepRewards):
        for outcome in rewards:
            name = outcome.name
            value = outcome.value

            # avoid isinstance check in most cases
            if value > 0:
                self._add_to_dict(self._curr_rewards, name, value)
            elif value < 0:
                self._add_to_dict(self._curr_punishments, name, value)
            else:
                if isinstance(value, Reward):
                    self._add_to_dict(self._curr_rewards, name, value)
                else:
                    self._add_to_dict(self._curr_rewards, name, value)

    def end_scenario(self, terminated, truncated, reason):
        for key, count_total in self._curr_rewards.items():
            self._rewards.setdefault(key, []).append(count_total)

        for key, count_total in self._curr_punishments.items():
            self._punishments.setdefault(key, []).append(count_total)

        self._curr_rewards.clear()
        self._curr_punishments.clear()

    def enumerate_values(self):
        yield from self._enumerate_dict(self._rewards, "rewards")
        yield from self._enumerate_dict(self._punishments, "punishments")

    def _enumerate_dict(self, dictionary, dictionary_name):
        for name, count_total_list in dictionary.items():
            if len(count_total_list) == 0:
                yield f"{dictionary_name}/{name}", 0.0
                yield f"{dictionary_name}-count/{name}", 0
            else:
                yield f"{dictionary_name}/{name}", sum(x.total for x in count_total_list) / len(count_total_list)
                yield f"{dictionary_name}-count/{name}", sum(x.count for x in count_total_list) / len(count_total_list)

    def _add_to_dict(self, dictionary, key, value, count = 1):
        if key not in dictionary:
            counter = dictionary[key] = RewardDetailsMetric.CountTotal()
        else:
            counter = dictionary[key]

        counter.add(value, count)

    class CountTotal:
        """Tracks the total count and value."""
        def __init__(self):
            self.total = 0
            self.count = 0

        def add(self, value, count = 1):
            """Adds a value."""
            self.total += value
            self.count += count

        def clear(self):
            """Clears the count."""
            self.total = 0
            self.count = 0

class EndingMetric(EnumMetric):
    """Tracks the ending of the scenario."""
    def __init__(self):
        super().__init__("endings", True, False)

    def end_scenario(self, terminated, truncated, reason):
        self.add(reason)

METRICS = {
    "overworld-progress" : lambda: RoomProgressMetric({
            (0, 0x77) : 0,
            (0, 0x67) : 1,
            (0, 0x78) : 1,
            (0, 0x68) : 2,
            (0, 0x58) : 3,
            (0, 0x48) : 4,
            (0, 0x38) : 5,
            (0, 0x37) : 6,
            (1, 0x73) : 7
        }),

    "dungeon1-progress" : lambda: RoomProgressMetric({
            (1, 0x72) : 1,
            (1, 0x74) : 1,
            (1, 0x63) : 3,
            (1, 0x53) : 4,
            (1, 0x52) : 5,
            (1, 0x42) : 6,
            (1, 0x43) : 7,
            (1, 0x44) : 8,
            (1, 0x45) : 9,
            (1, 0x35) : 10,
            (1, 0x36) : 11,
        }),

    "room-result" : RoomResultMetric,
    "room-health" : RoomHealthChangeMetric,
    "success-rate" : SuccessMetric,
    "reward-average" : RewardAverageMetric,
    "reward-details" : RewardDetailsMetric,
    "ending" : EndingMetric,
}

class MetricTracker:
    """Singleton class that tracks all metrics."""
    _instance : 'MetricTracker' = None

    @staticmethod
    def get_instance():
        """Returns the singleton instance."""
        return MetricTracker._instance

    def __init__(self, metric_names):
        self.metrics = [METRICS[name]() for name in metric_names]

        assert MetricTracker._instance is None
        MetricTracker._instance = self

    @staticmethod
    def close():
        """Closes the metric tracker."""
        MetricTracker._instance = None

    def begin_scenario(self, state):
        """Begins a new scenario."""
        for metric in self.metrics:
            metric.begin_scenario(state)

    def step(self, state_change : StateChange, rewards : StepRewards):
        """Steps the metrics."""
        for metric in self.metrics:
            metric.step(state_change, rewards)

    def end_scenario(self, terminated, truncated, reason):
        """Ends a scenario."""
        for metric in self.metrics:
            metric.end_scenario(terminated, truncated, reason)

    def get_progress_metric(self):
        """Returns the RoomProgressMetric instance, if any."""
        for metric in self.metrics:
            if isinstance(metric, RoomProgressMetric):
                return metric
        return None

    def get_metrics(self):
        """Enumerates the values of the metrics."""
        result = {}
        for metric in self.metrics:
            for key, value in metric.enumerate_values():
                result[key] = value

        return result

    @staticmethod
    def get_metrics_and_clear():
        """Enumerates the values of the metrics and clears them."""
        instance = MetricTracker._instance
        if instance is None:
            return {}

        result = instance.get_metrics()
        for metric in instance.metrics:
            metric.clear()

        return result
