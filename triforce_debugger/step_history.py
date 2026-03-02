"""Step history ring buffer for time-travel debugging.

Stores step snapshots in a fixed-capacity deque so the user can click any
past step and have all panels update to that step's data.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class StepEntry:
    """A single snapshot stored in the step history."""
    step_number: int
    action: Any               # ActionTaken
    reward: Any               # StepRewards
    observation: Any          # obs dict (tensors)
    state: Any                # ZeldaGame
    action_mask: Any          # Tensor
    action_probabilities: Any # OrderedDict from model
    terminated: bool
    truncated: bool
    frame: Any                # RGB numpy array (last game frame)


class StepHistory:
    """Ring buffer storing step snapshots for time-travel debugging.

    Backed by ``collections.deque(maxlen=MAX_STEPS)``.  Oldest entries are
    silently evicted when the buffer is full.
    """

    MAX_STEPS = 50_000

    def __init__(self, maxlen: Optional[int] = None):
        capacity = maxlen if maxlen is not None else self.MAX_STEPS
        self._buffer: deque[StepEntry] = deque(maxlen=capacity)

    # -- mutators ----------------------------------------------------------

    def append(self, entry: StepEntry) -> None:
        """Append a step entry, evicting the oldest if at capacity."""
        self._buffer.append(entry)

    def clear(self) -> None:
        """Remove all entries (e.g. on episode reset)."""
        self._buffer.clear()

    # -- accessors ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, index: int) -> StepEntry:
        """Retrieve by position (0 = oldest, -1 = newest)."""
        return self._buffer[index]

    def get_by_index(self, index: int) -> StepEntry:
        """Retrieve by position (0 = oldest, -1 = newest).

        Raises IndexError if out of range.
        """
        return self._buffer[index]

    @property
    def newest(self) -> Optional[StepEntry]:
        """Return the most recent entry, or None if empty."""
        return self._buffer[-1] if self._buffer else None

    @property
    def oldest(self) -> Optional[StepEntry]:
        """Return the oldest entry, or None if empty."""
        return self._buffer[0] if self._buffer else None
