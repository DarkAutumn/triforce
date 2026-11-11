"""The golden path: one run from the route's start to its last triforce, stitched from races.

The route is cut at its checkpoints: the overworld rooms Link stands in right after taking a triforce that is not
the route's last (``checkpoints``; d2first: route steps 24 and 68, back outside Dungeons 2 and 1).  For each segment
in turn, ``parallel`` games race from the same state to the segment's stop (a ``ChainPool`` of the segment's
seeds, every run recorded): the first run to get there wins, its end state becomes the checkpoint every game of
the next segment starts from, and the other games are dropped.  The winners' replays, joined
(``_native.replay_stitch``), are one replay from the start to the route's end: the golden path.

The games of a segment differ only by the models' sampling noise (keyed to the run seed): nothing is written to
the machine and no frame is inserted at a checkpoint, so the stitched replay plays straight through, as one
game.  (A power-on start with a random delay also gives every game of the first segment its own delay.)  What
a golden replay's EDITS hold is what any chain run's do: the action executor's writes (Link's facing ``$98``
before an attack, the selected B item ``$656``) and cheat grants; from power on there is no other write, and
none to the game's random numbers.  A segment nobody wins in ``attempts`` runs ends the path as a failure, with
the run that got furthest as its last part.
"""

from typing import Callable, List, Optional, Tuple

from . import _native
from .runner import DEFAULT_MAX_STEPS, ChainPool, Policies, Route, RunnerError

DEFAULT_ATTEMPTS = 320      # runs per segment before the path fails
POLL_SECONDS = 0.05


def checkpoints(route_name: str) -> List[int]:
    """The route steps entered right after a triforce that is not the route's last step (d2first: [24, 68])."""
    steps = _native.route_steps(route_name)
    return [step['index'] + 1 for step in steps[:-1] if step['exit'] == 'triforce']


def race(route: Route, seeds, policies: Policies, parallel: int, record: dict, *, threads: Optional[int] = None,
         states_dir=None, deterministic: Optional[bool] = False,
         max_steps: int = DEFAULT_MAX_STEPS) -> Tuple[dict, bytes, int]:
    """Plays ``route`` from ``seeds`` on ``parallel`` games until one run succeeds.  Returns (the chosen run's
    result dict, its recording, the runs that finished).  The winner is the success with the fewest frames, then
    the lowest seed, among those finished when the first success is seen; the other games are dropped.  Without
    a success: the failed run with the most rooms, then the most frames, then the lowest seed.  ``RunnerError``
    when every run ended in an error (e.g. a specialist without a model)."""
    # pylint: disable=too-many-arguments, too-many-locals
    results, recordings, wins = {}, {}, []
    with ChainPool(route, seeds, policies, num_envs=parallel, threads=threads, deterministic=deterministic,
                   max_steps=max_steps, record=record, states_dir=states_dir) as pool:
        pool.play()
        idle = False
        while not idle and not wins:
            idle = pool.pool.wait_idle(timeout=POLL_SECONDS)
            for event in pool.poll_events():
                if event['type'] == 'run':
                    results[int(event['result']['seed'])] = event['result']
                elif event['type'] == 'replay':
                    recordings[int(event['job'])] = event['data']
            wins = [seed for seed, result in results.items() if result['result'] == 'success' and seed in recordings]
    if wins:
        seed = min(wins, key=lambda s: (results[s]['frames'], s))
    else:
        played = [seed for seed, result in results.items() if result['result'] != 'error' and seed in recordings]
        if not played:
            reasons = sorted({result['reason'] for result in results.values()})
            raise RunnerError(f"no run of route '{route.name}' played: {'; '.join(reasons) or 'no seeds'}")
        seed = max(played, key=lambda s: (len(results[s]['rooms']), results[s]['frames'], -s))
    return results[seed], recordings[seed], len(results)


def golden_path(policies: Policies, route_name: str, parallel: int, first_seed: int, record: dict, *,
                start_from: Optional[str] = None, power_on: Optional[dict] = None,
                attempts: int = DEFAULT_ATTEMPTS, stops: Optional[list] = None, states_dir=None,
                threads: Optional[int] = None, on_segment: Optional[Callable[[dict], None]] = None,
                deterministic: Optional[bool] = False, max_steps: int = DEFAULT_MAX_STEPS) -> Tuple[bytes, dict]:
    """Plays the golden path of ``route_name`` (module doc) and returns (the stitched replay, a summary).

    ``record``: ``replay.record_options`` of every run; ``start_from`` / ``power_on``: the first segment's start
    (``Route.load``; neither: the game start state); ``stops``: the route step that ends each segment, None for
    the route's own end (default: the ``checkpoints`` after the start's route step, then None); segment ``s``
    plays seeds ``first_seed + s * attempts`` onward, at most ``attempts`` of them; ``deterministic``: how the
    models act (False: all sample, the default; None: each model's registered mode; True makes every game of a
    segment the same); ``on_segment(entry)`` is called with each segment's summary entry as it ends.

    Summary: ``{'result': 'success' | 'failure', 'frames', 'segments': [{'index', 'stop_step', 'seed', 'frames',
    'runs_played', 'ending'}]}`` (``frames``: the replay's; a segment's: its own replay's)."""
    # pylint: disable=too-many-arguments, too-many-locals
    if stops is None:
        start = Route.load(route_name, start_from=start_from, states_dir=states_dir, power_on=power_on).start_step
        stops = [step for step in checkpoints(route_name) if step > start] + [None]
    stops = list(stops)
    if not stops:
        raise ValueError("the golden path needs at least one segment")
    data: Optional[bytes] = None
    checkpoint = None
    segments = []
    for index, stop in enumerate(stops):
        if index == 0:
            route = Route.load(route_name, start_from=start_from, states_dir=states_dir, stop_step=stop,
                               power_on=power_on)
        else:
            route = Route.load(route_name, states_dir=states_dir, stop_step=stop, checkpoint=checkpoint)
        seeds = range(first_seed + index * attempts, first_seed + (index + 1) * attempts)
        result, recording, played = race(route, seeds, policies, parallel, record, threads=threads,
                                         states_dir=states_dir, deterministic=deterministic, max_steps=max_steps)
        segment = _native.Replay(recording)
        entry = {'index': index, 'stop_step': stop, 'seed': int(result['seed']), 'frames': segment.frames,
                 'runs_played': played, 'ending': segment.ending}
        segments.append(entry)
        if on_segment is not None:
            on_segment(entry)
        data = recording if data is None else _native.replay_stitch(data, recording)
        if result['result'] != 'success':
            break
        checkpoint = segment.states()[-1]
    won = len(segments) == len(stops) and segments[-1]['ending'] == 'win'
    return data, {'result': 'success' if won else 'failure', 'frames': _native.Replay(data).frames,
                  'segments': segments}
