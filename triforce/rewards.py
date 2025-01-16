from dataclasses import dataclass
import time
from typing import Iterator
import gymnasium as gym

REWARD_MINIMUM = 0.01
REWARD_TINY = 0.05
REWARD_SMALL = 0.25
REWARD_MEDIUM = 0.5
REWARD_LARGE = 0.75
REWARD_MAXIMUM = 1.0

@dataclass(frozen=True)
class Outcome:
    """Base class for outcomes."""
    name: str
    value: float
    count: int = 1

    def __radd__(self, other):
        # Support reversed addition (e.g., sum() starts with 0)
        if isinstance(other, (int, float)):
            return self.value + other

        if isinstance(other, Outcome):
            return self.value + other.value

        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Outcome):
            other = other.value

        return self.value < other

    def __le__(self, other):
        if isinstance(other, Outcome):
            other = other.value

        return self.value <= other

    def __gt__(self, other):
        if isinstance(other, Outcome):
            other = other.value

        return self.value > other

    def __ge__(self, other):
        if isinstance(other, Outcome):
            other = other.value

        return self.value >= other

    def copy(self):
        """Return a copy of the outcome."""
        if isinstance(self, Reward):
            return Reward(self.name, self.value, self.count)
        if isinstance(self, Penalty):
            return Penalty(self.name, self.value, self.count)
        raise ValueError(f"Unknown outcome type: {self}")

    def merge(self, other : 'Outcome'):
        """Merge two outcomes."""
        assert self.name == other.name
        if isinstance(self, Reward):
            return Reward(self.name, self.value + other.value, self.count + other.count)
        if isinstance(self, Penalty):
            return Penalty(self.name, self.value + other.value, self.count + other.count)
        raise ValueError(f"Unknown outcome type: {self}")

@dataclass(frozen=True)
class Reward(Outcome):
    """Represents a positive outcome (reward)."""
    def __post_init__(self):
        assert self.value >= 0
        assert self.name.startswith("reward")

@dataclass(frozen=True)
class Penalty(Outcome):
    """Represents a negative outcome (penalty)."""
    def __post_init__(self):
        assert self.value <= 0
        assert self.name.startswith("penalty")

class StepRewards:
    """A single step's rewards."""
    def __init__(self):
        self._history = []
        self._outcomes = {}
        self.ending = None
        self.score = None

    def __repr__(self):
        s = sorted(self._outcomes.values(), key=lambda x: x.value)
        inside = ", ".join(f"{x.name}={x.value}" for x in s)
        return f"{self.value} := {inside}"

    def __iter__(self) -> Iterator[Outcome]:
        """Make StepRewards iterable by delegating to the rewards list."""
        return iter(self._outcomes.values())

    def __len__(self):
        """Return the number of rewards."""
        return len(self._outcomes)

    def __contains__(self, key):
        return key in self._outcomes

    def __getitem__(self, key):
        return self._outcomes[key]

    @property
    def value(self):
        """The total reward value."""
        return max(min(sum(self._outcomes.values()), REWARD_MAXIMUM), -REWARD_MAXIMUM)

    def add(self, outcome: Outcome, scale=None):
        """Add an outcome to the rewards."""
        if scale is not None:
            value = max(min(outcome.value * scale, REWARD_MAXIMUM), -REWARD_MAXIMUM)
            if isinstance(outcome, Reward):
                outcome = Reward(outcome.name, value)
            elif isinstance(outcome, Penalty):
                outcome = Penalty(outcome.name, value)
            else:
                raise ValueError(f"Unknown outcome type: {outcome}")

        assert outcome.name not in self._outcomes, f"Duplicate outcome: {outcome.name}"
        self._outcomes[outcome.name] = outcome

    def remove_rewards(self):
        """Removes all rewards for this step."""
        self._outcomes = {x: y for x, y in self._outcomes.items() if isinstance(y, Penalty)}

class EpisodeRewards:
    """A running total of rewards for the current episode."""
    def __init__(self):
        self.rewards = 0
        self.score = 0
        self.outcomes = {}
        self.frames = 0
        self.steps = 0
        self.ending = None

    def update(self, frames, steps, rewards : StepRewards):
        """Update the running rewards with the rewards for the current frame."""
        assert not self.ending, "Cannot update a terminated episode"

        self.frames = frames
        self.steps = steps
        self.score = rewards.score
        self.rewards += rewards.value
        for outcome in rewards:
            if outcome.name not in self.outcomes:
                self.outcomes[outcome.name] = outcome.copy()
            else:
                self.outcomes[outcome.name] = self.outcomes[outcome.name].merge(outcome)

        if rewards.ending:
            self.ending = rewards.ending

class TotalRewards:
    """A wrapper that calculates the total reward for the episode."""
    _outcomes_seen = set()
    _endings_seen = set()

    def __init__(self):
        self.rewards = []
        self.scores = []
        self.total_steps = []
        self.outcomes = self._create_outcome_dict()
        self.endings = {x : 0 for x in TotalRewards._endings_seen}
        self.episodes = 0

    def _create_outcome_dict(self):
        result = {}
        for key in TotalRewards._outcomes_seen:
            if key.startswith("reward"):
                result[key] = Reward(key, 0)
            else:
                result[key] = Penalty(key, 0)

        return result

    def add(self, rewards : EpisodeRewards):
        """Add the rewards for the current episode."""
        assert rewards.ending, "Cannot add rewards for an episode that has not ended"

        self.rewards.append(rewards.rewards)
        self.scores.append(rewards.score)
        self.total_steps.append(rewards.steps)
        self.episodes += 1

        for outcome in rewards.outcomes.values():
            if outcome.name not in self.outcomes:
                self.outcomes[outcome.name] = outcome.copy()
                TotalRewards._outcomes_seen.add(outcome.name)
            else:
                self.outcomes[outcome.name] = self.outcomes[outcome.name].merge(outcome)

        if rewards.ending:
            self.endings[rewards.ending] = self.endings.get(rewards.ending, 0) + 1
            TotalRewards._endings_seen.add(rewards.ending)

    @property
    def stats(self):
        """Return the total rewards statistics."""
        return RewardStats(self)

    def get_stats_and_clear(self):
        """Clears this object, returning the stats up to that point."""
        stats = self.stats
        self.rewards.clear()
        self.scores.clear()
        self.total_steps.clear()
        self.outcomes = self._create_outcome_dict()
        self.endings = {x : 0 for x in TotalRewards._endings_seen}
        self.episodes = 0

        return stats

class RewardStats:
    """Totalled rewards for a section of a training run."""
    def __init__(self, total : TotalRewards, evaluated = False):
        self.evaluated = evaluated
        self.episodes = total.episodes
        self.reward_mean = self._mean(total.rewards)
        self.progress_mean = self._mean(total.scores)
        self.total_steps = self._mean(total.total_steps)
        self.outcomes = {x: y.copy() for x, y in total.outcomes.items()}
        self.endings = total.endings.copy()

    def _mean(self, values):
        if not values:
            return 0

        return sum(values) / len(values)

    @property
    def success_rate(self):
        """Return the success rate."""
        if not self.episodes:
            return 0

        successes = sum(value for key, value in self.endings.items() if key.startswith('success'))
        return successes / self.episodes


    def to_tensorboard(self, tensorboard, iterations):
        """Write the stats to TensorBoard."""
        if not tensorboard:
            return

        curr = time.time()

        tensorboard.add_scalar('evaluation/success-rate', self.success_rate, iterations, curr)
        tensorboard.add_scalar('evaluation/ep-reward-avg', self.reward_mean, iterations, curr)
        tensorboard.add_scalar('evaluation/progress', self.progress_mean, iterations, curr)
        tensorboard.add_scalar('rollout/steps-per-episode', self.total_steps, iterations, curr)
        tensorboard.add_scalar('rollout/seconds-per-episode', self.total_steps / 60.1, iterations, curr)

        for reward in (o for o in self.outcomes.values() if isinstance(o, Reward)):
            tensorboard.add_scalar(f'rewards/{reward.name}', reward.value, iterations, curr)
            tensorboard.add_scalar(f'reward-counts/{reward.name}', reward.count, iterations, curr)

        for penalty in (o for o in self.outcomes.values() if isinstance(o, Penalty)):
            tensorboard.add_scalar(f'penalties/{penalty.name}', penalty.value, iterations, curr)
            tensorboard.add_scalar(f'penalty-counts/{penalty.name}', penalty.count, iterations, curr)

        if self.episodes and self.endings:
            for name, count in self.endings.items():
                tensorboard.add_scalar(f'endings/{name}', count / self.episodes, iterations, curr)

        tensorboard.flush()

class EpisodeRewardTracker(gym.Wrapper):
    """A wrapper that tracks rewards for the current episode."""
    def __init__(self, env):
        super().__init__(env)

        self._episode_rewards = None

    def reset(self, **kwargs):
        self._episode_rewards = EpisodeRewards()
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, rewards, terminated, truncated, state_change = self.env.step(action)
        self._update(rewards, state_change, terminated or truncated)
        return observation, rewards, terminated, truncated, state_change

    def _update(self, rewards : StepRewards, state_change, done):
        assert done == bool(rewards.ending), "Reward ending and done state do not match"

        info = state_change.state.info
        frames = info['total_frames']
        steps = info['steps']
        self._episode_rewards.update(frames, steps, rewards)

        if rewards.ending:
            info['episode_rewards'] = self._episode_rewards
            self._episode_rewards = None
