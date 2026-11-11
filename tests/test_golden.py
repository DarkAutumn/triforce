"""The golden path (``triforce.golden``): races from checkpoint to checkpoint, the winners stitched into one
replay."""

import pytest

from triforce import _native, golden, replay
from triforce.runner import Policies, Route, Runner, RunnerError, chain_meta

from .conftest import SEGMENT_START_STEP, SEGMENT_STATE, SEGMENT_STOP_STEP, require_state


def options(policies, route_name='d2first'):
    """Record options of golden-path runs played by ``policies`` (sampling)."""
    return replay.record_options(chain_meta(policies, False, Route.load(route_name)))


def test_checkpoints_follow_the_route_triforces():
    steps = _native.route_steps('d2first')
    assert golden.checkpoints('d2first') == [24, 68]
    # Back on the overworld outside Dungeon 2 and Dungeon 1; the route's last triforce is no checkpoint.
    assert [(steps[k]['level'], steps[k]['room']) for k in golden.checkpoints('d2first')] == [(0, 0x3c), (0, 0x37)]
    assert steps[-1]['exit'] == 'triforce'


def test_a_scripted_segment_is_won_and_recorded(scripted_models, states_dir):
    policies = Policies(scripted_models)
    entries = []
    data, summary = golden.golden_path(policies, 'd2first', 2, 10, options(policies),
                                       start_from=str(require_state(SEGMENT_STATE)), stops=[SEGMENT_STOP_STEP],
                                       states_dir=states_dir, attempts=4, on_segment=entries.append)
    recording = _native.Replay(data)
    assert summary == {'result': 'success', 'frames': recording.frames, 'segments': entries}
    segment, = entries
    assert (segment['index'], segment['stop_step'], segment['ending']) == (0, SEGMENT_STOP_STEP, 'win')
    assert 10 <= segment['seed'] < 14 and 1 <= segment['runs_played'] <= 4
    assert recording.ending == 'win' and recording.states()[-1]['step'] == SEGMENT_STOP_STEP
    assert _native.replay_verify(data)['frames'] == recording.frames


def test_segments_start_from_the_winner_and_a_lost_segment_ends_the_path(scripted_models, states_dir):
    """Two real segments: the scripted one is won (7 steps); nobody gets from its end to route step 68 in twelve
    steps, so the path fails there with the run that got furthest, stitched onto the winner."""
    policies = Policies(scripted_models)
    data, summary = golden.golden_path(policies, 'd2first', 2, 0, options(policies),
                                       start_from=str(require_state(SEGMENT_STATE)), stops=[SEGMENT_STOP_STEP, 68],
                                       states_dir=states_dir, attempts=3, max_steps=12)
    first, second = summary['segments']
    assert summary['result'] == 'failure'
    assert first['ending'] == 'win' and second['ending'] != 'win' and second['runs_played'] == 3
    assert 0 <= first['seed'] < 3 <= second['seed'] < 6, "each segment plays its own seeds"
    joined = _native.Replay(data)
    assert (joined.frames, joined.ending) == (first['frames'] + second['frames'], second['ending'])
    assert _native.replay_verify(data)['frames'] == joined.frames
    states = joined.states()
    assert [state['kind'] for state in states] == ['start', 'checkpoint', 'end']
    assert (states[1]['frame'], states[1]['step']) == (first['frames'], SEGMENT_STOP_STEP)
    rooms = [(event['room'], event['step']) for event in joined.events() if event['type'] == 'room']
    assert rooms[:3] == [(0x3c, SEGMENT_START_STEP), (0x4c, SEGMENT_STOP_STEP), (0x4c, SEGMENT_STOP_STEP)]
    assert len(joined.meta['segments']) == 2


def test_a_golden_path_from_power_on_is_one_power_on_replay(scripted_models, states_dir):
    policies = Policies(scripted_models)
    data, summary = golden.golden_path(policies, 'd2first', 2, 0, options(policies),
                                       power_on={'name': 'LINK', 'delay': 3}, stops=[3], states_dir=states_dir,
                                       attempts=2, max_steps=30)
    recording = _native.Replay(data)
    assert recording.source == {'kind': 'power-on', 'name': 'LINK', 'delay': 3}
    assert (summary['result'] == 'success') == (recording.ending == 'win')
    assert summary['segments'][0]['ending'] == recording.ending and summary['frames'] == recording.frames
    assert _native.replay_verify(data)['frames'] == recording.frames


def test_no_playable_run_is_an_error_not_a_path(states_dir, tmp_path):
    policies = Policies(tmp_path)           # no models at all
    with pytest.raises(RunnerError, match="no run of route 'd2first' played: .*ow-start"):
        golden.golden_path(policies, 'd2first', 2, 0, options(policies), stops=[3], states_dir=states_dir,
                           attempts=2)
    with pytest.raises(ValueError, match='at least one segment'):
        golden.golden_path(policies, 'd2first', 2, 0, options(policies), stops=[])


class FakePool:
    """A stand-in ``ChainPool`` serving canned events: every seed's run and recording in the first poll."""
    scripts: list = []          # per pool built: {seed: (result dict, recording bytes)}
    built: list = []            # (route, seeds) of every pool built

    def __init__(self, route, seeds, policies, **kwargs):
        self.script = FakePool.scripts[len(FakePool.built)]
        FakePool.built.append((route, list(seeds), kwargs))
        self.pool = self
        self.polled = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def play(self):
        return None

    def wait_idle(self, timeout=None):
        return self.polled

    def poll_events(self):
        if self.polled:
            return []
        self.polled = True
        events = []
        for seed, (result, data) in self.script.items():
            events.append({'type': 'replay', 'env': 0, 'job': seed, 'data': data})
            events.append({'type': 'run', 'env': 0, 'result': {**result, 'seed': seed}})
        return events


@pytest.fixture
def stitchable(scripted_models, states_dir):
    """(first, second): the scripted segment's winning replay and a replay continued from its end state."""
    runner = Runner(models_dir=scripted_models, max_steps=12, states_dir=states_dir)
    record = options(runner.policies)

    def play(route, seed):
        with runner.pool(route, [seed], record=record) as chain:
            chain.play()
            chain.pool.wait_idle()
            return next(event['data'] for event in chain.poll_events() if event['type'] == 'replay')

    first = play(Route.load('d2first', start_from=str(require_state(SEGMENT_STATE)), stop_step=SEGMENT_STOP_STEP), 0)
    second = play(Route.load('d2first', checkpoint=_native.Replay(first).states()[-1]), 1)
    return first, second


def test_the_fastest_winner_of_a_poll_becomes_the_next_checkpoint(stitchable, monkeypatch):
    first, second = stitchable
    won = {'result': 'success', 'reason': 'reached', 'rooms': [{}, {}]}
    lost = {'result': 'failure', 'reason': 'died', 'rooms': [{}]}
    FakePool.built = []
    FakePool.scripts = [
        # Two wins in one poll: the one with fewer frames is the segment's, whatever its seed; a failure isn't.
        {3: ({**won, 'frames': 500}, b'slower'), 9: ({**won, 'frames': 400}, first),
         5: ({**lost, 'frames': 100}, b'lost')},
        {21: ({**won, 'frames': 700}, second)},
    ]
    monkeypatch.setattr(golden, 'ChainPool', FakePool)
    data, summary = golden.golden_path(object(), 'd2first', 4, 0, {'record': True},
                                       start_from=str(require_state(SEGMENT_STATE)),
                                       stops=[SEGMENT_STOP_STEP, None], attempts=20, threads=3)
    assert data == _native.replay_stitch(first, second)
    assert [(s['index'], s['stop_step'], s['seed'], s['runs_played']) for s in summary['segments']] == [
        (0, SEGMENT_STOP_STEP, 9, 3), (1, None, 21, 1)]
    (route_a, seeds_a, kwargs_a), (route_b, seeds_b, _) = FakePool.built
    assert (seeds_a, seeds_b) == (list(range(0, 20)), list(range(20, 40)))
    assert (route_a.stop_step, route_a.checkpoint, route_b.stop_step) == (SEGMENT_STOP_STEP, None, None)
    end = _native.Replay(first).states()[-1]
    assert route_b.checkpoint == {'state': end['state'], 'step': SEGMENT_STOP_STEP, 'granted': end['granted']}
    assert (kwargs_a['num_envs'], kwargs_a['threads'], kwargs_a['deterministic'], kwargs_a['record']) == (
        4, 3, False, {'record': True})


def test_a_segment_without_a_win_keeps_the_run_that_got_furthest(stitchable, monkeypatch):
    first, _ = stitchable
    lost = {'result': 'failure', 'reason': 'died'}
    FakePool.built = []
    FakePool.scripts = [{
        4: ({**lost, 'frames': 900, 'rooms': [{}]}, b'one room'),
        6: ({**lost, 'frames': 300, 'rooms': [{}, {}, {}]}, first),         # the most rooms
        7: ({**lost, 'frames': 200, 'rooms': [{}, {}, {}]}, b'fewer frames'),
        8: ({'result': 'error', 'reason': 'no model', 'frames': 0, 'rooms': []}, b''),
    }]
    del FakePool.scripts[0][8]      # a run that never played has a result and no recording
    monkeypatch.setattr(golden, 'ChainPool', FakePool)
    data, summary = golden.golden_path(object(), 'd2first', 2, 0, {}, start_from=str(require_state(SEGMENT_STATE)),
                                       stops=[SEGMENT_STOP_STEP, None], attempts=5)
    assert data == first and len(FakePool.built) == 1, "the path ends with the lost segment"
    assert summary['result'] == 'failure' and [s['seed'] for s in summary['segments']] == [6]
