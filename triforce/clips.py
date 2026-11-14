"""The run-time clips switch on the Python side (round 22).

A family's ``clips:`` (``env::config::ClipsConfig``: ``enabled`` and ``off_share``, the share of training episodes
drawn with the switch off; ``DEFAULT_CLIPS_OFF_SHARE``, 0.5, for a clip family) decides what an evaluation plays:
``default_mode`` is ``both`` for a family with clips enabled (every start state with the switch on and with it off,
as far as the state allows) and ``on`` for the others.  A start state the route enters on one way of a stretch only
fixes the switch (``zelda::objectives::start_clips``: 4:0x21 from S the clip way, 4:0x30 from E the plain way):
``start_clips`` asks the env, and ``episode_jobs`` plays such a state only in the switch state it allows.
``label`` names a (start state, switch) pair the way the early stop and the evaluation's ``by_clips`` read it.
"""

from pathlib import Path
from typing import Iterable, Optional

from . import _native

MODES = ('on', 'off', 'both')
OFF_LABEL = ':clips-off'     # appended to a start state played with the switch off (training's per-state rates)
_FIXED: dict[tuple, dict[str, Optional[bool]]] = {}


def name(clips: Optional[bool]) -> str:
    """``on`` / ``off`` (``on`` for records from before the switch, which played with clips allowed)."""
    return 'off' if clips is False else 'on'


def label(state: Optional[str], clips: Optional[bool]) -> Optional[str]:
    """``state`` for the switch on (or unknown), ``state:clips-off`` for it off."""
    return f'{state}{OFF_LABEL}' if clips is False and state is not None else state


def default_mode(config) -> str:
    """``both`` for a family with clips enabled, else ``on``."""
    return 'both' if config.clips.enabled else 'on'


def switches(mode: str) -> tuple[bool, ...]:
    """The switch states a mode plays: ``on`` (True,), ``off`` (False,), ``both`` (True, False)."""
    if mode not in MODES:
        raise ValueError(f"clips mode {mode!r} is not one of {MODES}")
    return {'on': (True,), 'off': (False,), 'both': (True, False)}[mode]


def start_clips(config, states: Iterable[str], states_dir=None) -> dict[str, Optional[bool]]:
    """Start state -> the switch it fixes (True / False), None when either way is possible: one reset per state of
    an ``_native.Env`` (the reset info's ``clips_fixed``), cached per (family, states dir, state)."""
    states = list(states)
    key = (config.to_json(), None if states_dir is None else str(states_dir))
    cache = _FIXED.setdefault(key, {})
    missing = [state for state in states if state not in cache]
    if missing:
        env = _native.Env(config, explicit_starts=True, states_dir=states_dir)
        for state in missing:
            _, info = env.reset(seed=0, options={'start_state': state})
            cache[state] = info.get('clips_fixed')
    return {state: cache[state] for state in states}


def episode_jobs(config, episodes_per_state: int, seed: int = 0, mode: Optional[str] = None,
                 states_dir=None) -> list[tuple[str, int, bool]]:
    """``(start state, seed, switch)`` evaluation jobs: ``episodes_per_state`` episodes of every seed start state
    (sorted) for each switch state ``mode`` (default ``default_mode``) plays and the state allows, the switch on
    first; episode ``i`` seeded ``seed + i``, so the switch-on half of a ``both`` evaluation is the ``on``
    evaluation.  Raises ValueError when no start state allows the mode."""
    mode = default_mode(config) if mode is None else mode
    states_dir = None if states_dir is None else str(states_dir)
    states = sorted(Path(p).stem for p in config.resolve_start_states(states_dir))
    if not states:
        raise ValueError(f"{config.name} has no seed start states")
    fixed = start_clips(config, states, states_dir)
    plan = [(state, on) for on in switches(mode) for state in states if fixed[state] in (None, on)]
    if not plan:
        raise ValueError(f"{config.name}: no start state allows clips {mode}")
    episodes = [(state, on) for state, on in plan for _ in range(episodes_per_state)]
    return [(state, seed + index, on) for index, (state, on) in enumerate(episodes)]
