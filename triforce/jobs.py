"""Process-level settings of the training and evaluation commands: their CPU priority.

Training and evaluation run at a lower CPU priority so other work on the machine goes first: every training and
evaluation entry point (``scripts/train_round.py``, ``python -m triforce.train``, ``triforce evaluate``,
``triforce eval-chain``, ``triforce eval-batch``) calls ``lower_priority`` at start with its ``--nice N`` option
(default ``$TRIFORCE_NICE``, else ``DEFAULT_NICE``; 0 leaves the priority alone).  The niceness only ever goes up
(lower priority), never down: a process already at N or lower priority stays where it is, and children inherit it.
Library code never calls it, so play and replay tools (play-zelda, zelda-headless, create-video, zelda-map,
``tf replay``, explore, entry_states) that share the evaluation code keep their normal priority.  I/O priority is
left alone.
"""

import os
import sys
from typing import Optional

NICE_ENV = 'TRIFORCE_NICE'
DEFAULT_NICE = 15
NICE_HELP = (f"lower this process's CPU priority to niceness N (default ${NICE_ENV}, else {DEFAULT_NICE}; 0 leaves "
             "it alone; it is never raised)")


def default_nice() -> int:
    """``$TRIFORCE_NICE`` (an integer), else ``DEFAULT_NICE``."""
    value = os.environ.get(NICE_ENV, '').strip()
    if not value:
        return DEFAULT_NICE
    try:
        return int(value)
    except ValueError:
        print(f"triforce: warning: ${NICE_ENV}={value!r} is not an integer; using {DEFAULT_NICE}", file=sys.stderr)
        return DEFAULT_NICE


def add_nice_argument(parser):
    """The ``--nice N`` option of a training or evaluation command (default None: ``default_nice()``)."""
    parser.add_argument('--nice', type=int, default=None, metavar='N', help=NICE_HELP)


def lower_priority(nice: Optional[int] = None) -> int:
    """Raises this process's niceness to ``nice`` (None: ``default_nice()``) when it is lower; never lowers it (0
    or a value at or below the current niceness changes nothing).  Returns the niceness the process runs at."""
    target = default_nice() if nice is None else nice
    current = os.nice(0)
    if target > current:
        try:
            current = os.nice(target - current)
        except OSError as error:
            print(f"triforce: warning: could not lower the priority to nice {target}: {error}", file=sys.stderr)
    return current


def child_env(nice: Optional[int] = None, env: Optional[dict] = None) -> dict:
    """The environment of a child training or evaluation: ``$TRIFORCE_NICE`` set to the parent's niceness (``nice``,
    None: this process's), so the child's own ``lower_priority`` keeps what the parent chose (also ``--nice 0``)."""
    return {**(os.environ if env is None else env), NICE_ENV: str(os.nice(0) if nice is None else nice)}
