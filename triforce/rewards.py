
from dataclasses import dataclass
from typing import Iterator

import numpy as np

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

@dataclass(frozen=True)
class Reward(Outcome):
    """Represents a positive outcome (reward)."""
    def __post_init__(self):
        assert self.value >= 0
        assert self.value <= REWARD_MAXIMUM

@dataclass(frozen=True)
class Penalty(Outcome):
    """Represents a negative outcome (penalty)."""
    def __post_init__(self):
        assert self.value <= 0
        assert self.value >= -REWARD_MAXIMUM

class StepRewards:
    """A single step's rewards."""
    def __init__(self):
        self._outcomes = {}
        self.ending = None
        self.score = None

    def __repr__(self):
        s = sorted(self._outcomes.values(), key=lambda x: x.value)
        inside = ", ".join(f"{x.name}={x.value}" for x in s)
        return f"{self.value} := {inside}"

    def __iter__(self) -> Iterator[Outcome]:
        """Make StepRewards iterable by delegating to the rewards list."""
        return self._outcomes.values()

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
        return np.clip(sum(self._outcomes.values()), -REWARD_MAXIMUM, REWARD_MAXIMUM)

    def add(self, outcome: Outcome, scale=None):
        """Add an outcome to the rewards."""
        if scale is not None:
            value = np.clip(outcome.value * scale, -REWARD_MAXIMUM, REWARD_MAXIMUM)
            outcome = Outcome(outcome.name, value)

        assert outcome.name not in self._outcomes, f"Duplicate outcome: {outcome.name}"
        self._outcomes[outcome.name] = outcome

    def zero_rewards(self):
        """Zero out all rewards, but keep the entry so we know it was there."""
        self._outcomes = {x: Reward(x, 0) if isinstance(x, Reward) else self._outcomes[x] for x in self._outcomes}
