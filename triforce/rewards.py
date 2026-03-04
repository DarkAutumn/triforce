from dataclasses import dataclass
from typing import Iterator

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
