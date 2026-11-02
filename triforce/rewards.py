"""Named reward/penalty outcomes, per-step clamping and terminal rewards (ported from v1, design §6)."""

from dataclasses import dataclass
from typing import Iterator, Optional

REWARD_MINIMUM = 0.01
REWARD_TINY = 0.05
REWARD_SMALL = 0.25
REWARD_MEDIUM = 0.5
REWARD_LARGE = 0.75
REWARD_MAXIMUM = 1.0
TERMINAL_REWARD_MIN = -20.0
TERMINAL_FAILURE_PENALTY_VALUE = 20.0
TERMINAL_SUCCESS_REWARD_VALUE = 20.0

# Terminal outcome classes (design §6: deny-by-default classification, see end_conditions.classify_ending).
SUCCESS = 'success'
HARD_FAILURE = 'hard-failure'
SOFT_FAILURE = 'soft-failure'
OUTCOMES = (SUCCESS, HARD_FAILURE, SOFT_FAILURE)

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
        self._outcomes = {}
        self.ending: Optional[str] = None      # end-condition name on the last step
        self.outcome: Optional[str] = None     # SUCCESS / HARD_FAILURE / SOFT_FAILURE on the last step

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

    def to_dict(self) -> dict:
        """name -> value for every outcome this step."""
        return {name: outcome.value for name, outcome in self._outcomes.items()}

    @property
    def value(self):
        """The total reward value, clamped to ±REWARD_MAXIMUM except for terminal rewards."""
        total = sum(self._outcomes.values())
        if self.outcome == SUCCESS and REWARD_TERMINAL_SUCCESS in self._outcomes:
            return max(min(total, TERMINAL_SUCCESS_REWARD_VALUE), -REWARD_MAXIMUM)
        if self.outcome in (HARD_FAILURE, SOFT_FAILURE) and PENALTY_TERMINAL_FAILURE in self._outcomes:
            return max(min(total, REWARD_MAXIMUM), TERMINAL_REWARD_MIN)
        return max(min(total, REWARD_MAXIMUM), -REWARD_MAXIMUM)

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

    def scale_rewards(self, factor, exempt_prefixes=None):
        """Scales all positive rewards by the given factor.

        Rewards whose names start with any of the exempt_prefixes are not scaled.
        Penalties are never scaled.
        """
        if exempt_prefixes is None:
            exempt_prefixes = ()
        scaled = {}
        for name, outcome in self._outcomes.items():
            if isinstance(outcome, Reward) and not any(name.startswith(p) for p in exempt_prefixes):
                scaled[name] = Reward(name, outcome.value * factor, outcome.count)
            else:
                scaled[name] = outcome
        self._outcomes = scaled


REWARD_TERMINAL_SUCCESS = "reward-terminal-success"
PENALTY_TERMINAL_FAILURE = "penalty-terminal-failure"


def success_reward_value(health: float, max_health: float) -> float:
    """Design §6: health is handed to the next specialist, so success pays +20 × (0.5 + 0.5 × hp/max_hp)."""
    frac = max(0.0, min(1.0, health / max_health)) if max_health > 0 else 0.0
    return TERMINAL_SUCCESS_REWARD_VALUE * (0.5 + 0.5 * frac)


def apply_terminal_rewards(rewards: StepRewards, ending: str, outcome: str, terminated: bool,
                           health: float, max_health: float) -> None:
    """Records the ending/outcome and adds the terminal reward (v1 apply_terminal_rewards semantics).

    - success: + health-scaled success reward.
    - hard failure: step rewards are dropped (penalties kept) and −20 is added.
    - soft failure: −20 is added, step rewards kept (v1's "other failure" endings).
    - truncation (terminated=False, e.g. max-steps): classified but no terminal reward, so the value
      function can bootstrap.
    """
    assert outcome in OUTCOMES, f"Unknown outcome {outcome!r}"
    rewards.ending = ending
    rewards.outcome = outcome
    if not terminated:
        return

    if outcome == SUCCESS:
        rewards.add(Reward(REWARD_TERMINAL_SUCCESS, success_reward_value(health, max_health)))
    else:
        if outcome == HARD_FAILURE:
            rewards.remove_rewards()
        rewards.add(Penalty(PENALTY_TERMINAL_FAILURE, -TERMINAL_FAILURE_PENALTY_VALUE))
