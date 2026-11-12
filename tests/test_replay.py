"""Recordings (``triforce.replay``, TFREPLAY files) of chain runs and evaluation episodes: one file per run that
replays on the emulator alone, its optional streams, its start (state file, embedded state, power on), and
cutting, continuing and stitching.  The file format itself is tested in ``crates/replay``."""

import json
import shutil

import numpy as np
import pytest

from triforce import _native, cli, registry, replay
from triforce.evaluate import evaluate_specialist
from triforce.runner import Route, Runner

from .conftest import REPO_STATES_DIR, SEGMENT_START_STEP, SEGMENT_STATE, SEGMENT_STOP_STEP, require_state

STREAMS = ('rewards', 'decisions', 'keyframes')
KINDS, DIRECTIONS, DURATIONS = 8, 8, 4
DISTRIBUTION = KINDS + KINDS * DIRECTIONS + KINDS * DURATIONS


def record(runner, route, seed, **options):
    """(the run's result, its recording) of one chain run recorded with ``replay.record_options(**options)``."""
    events = []
    with runner.pool(route, [seed], record=replay.record_options(runner.record_meta(route), **options)) as chain:
        chain.play()
        idle = False
        while not idle:
            idle = chain.pool.wait_idle(timeout=0.1)
            events += chain.poll_events()
    (_, result), = chain.results(events)
    recording, = [_native.Replay(event['data']) for event in events if event['type'] == 'replay']
    return result, recording


def end_of(recording, states_dir=None):
    """RAM after playing ``recording`` to its last frame on the emulator alone."""
    player = _native.ReplayPlayer(recording.data, states_dir=None if states_dir is None else str(states_dir))
    player.seek(player.frames)
    assert player.frame == recording.frames
    return player.ram()


@pytest.fixture(scope='module')
def chain_replays(states_dir, scripted_models, tmp_path_factory):
    """(folder, results, runner) of three runs from the game start (the untrained ow-start, 40 steps each), every
    stream recorded, a keyframe every 60 frames."""
    folder = tmp_path_factory.mktemp('chain-replays')
    runner = Runner(models_dir=scripted_models, max_steps=40, parallel=3, states_dir=states_dir)
    route = Route.load('d2first')
    options = replay.record_options(runner.record_meta(route), STREAMS, keyframe_frames=60)
    results = runner.run_many(route, [0, 1, 2], record=options, replays=replay.ReplayDir(folder))
    return folder, results, runner


def test_every_run_is_one_replay_file_that_replays_on_the_emulator_alone(chain_replays, states_dir):
    folder, results, _ = chain_replays
    files = sorted(folder.iterdir())
    assert len(files) == len(results) == 3
    for path, result in zip(files, results):
        recording = _native.Replay(path.read_bytes())
        assert path.name == f'{result.seed:08d}-{recording.ending}.tfreplay' and recording.ending in replay.ENDINGS
        meta = recording.meta
        assert (meta['seed'], meta['kind'], meta['route'], meta['stop_at']) == (result.seed, 'chain', 'd2first', None)
        assert (meta['ending']['result'], meta['ending']['reason']) == (result.result, result.reason)
        assert meta['ending']['outcome'] == result.rooms[-1].outcome
        assert [model['name'] for model in meta['models']] == sorted(model['name'] for model in meta['models'])
        assert (recording.version, recording.source) == (0, {'kind': 'state', 'name': 'start'})
        # The start state is embedded: the file replays without the states dir, to the state it recorded.
        report = _native.replay_verify(recording.data)
        assert (report['frames'], report['states']) == (recording.frames, len(recording.states()))
        assert np.array_equal(end_of(recording), recording.end_ram())
        rooms = [event for event in recording.events() if event['type'] == 'room']
        assert (rooms[0]['level'], rooms[0]['room'], rooms[0]['step']) == (0, 0x77, 0)
        assert len(rooms) == len(result.rooms)


def test_the_optional_streams_keep_rewards_decisions_and_keyframes(chain_replays):
    folder, results, _ = chain_replays
    for path, result in zip(sorted(folder.iterdir()), results):
        recording = _native.Replay(path.read_bytes())
        rewards = recording.rewards()
        assert rewards and {name for _, name, _ in rewards} <= set(_native.REWARD_TERMS)
        assert set(recording.meta['reward_terms']) == {name for _, name, _ in rewards}

        decisions = recording.decisions()
        dist, action = decisions['dist'], decisions['action']
        assert dist.shape == (result.steps, DISTRIBUTION) and recording.meta['actions']['distribution'] == DISTRIBUTION
        assert np.allclose(np.exp(dist[:, :KINDS]).sum(axis=1), 1, atol=1e-4)
        rows = np.arange(result.steps)
        kind, direction, duration = action.T
        taken = np.stack([dist[rows, kind], dist[rows, KINDS + kind * DIRECTIONS + direction],
                          dist[rows, KINDS + KINDS * DIRECTIONS + kind * DURATIONS + duration]])
        assert np.isfinite(taken).all() and (taken <= 1e-6).all(), "the taken action has a probability"
        assert np.isfinite(decisions['value']).all() and (np.diff(decisions['frame']) > 0).all()

        states = recording.states()
        keyframes = [state for state in states if state['kind'] == 'keyframe']
        assert [state['kind'] for state in states] == ['start'] + ['keyframe'] * len(keyframes) + ['end']
        assert keyframes and all(b['frame'] - a['frame'] >= 60 for a, b in zip(keyframes, keyframes[1:]))
        # Seeking past a keyframe starts from it; the machine is the one stepping there from the start reaches.
        target = keyframes[-1]['frame'] + 7
        seeked, stepped = _native.ReplayPlayer(recording.data), _native.ReplayPlayer(recording.data)
        seeked.seek(target)
        assert stepped.step(target) == target and seeked.frame == stepped.frame == target
        assert np.array_equal(seeked.ram(), stepped.ram())


def test_a_replay_that_names_its_state_needs_that_state_file(scripted_models, states_dir, tmp_path):
    runner = Runner(models_dir=scripted_models, max_steps=8, states_dir=states_dir)
    _, recording = record(runner, Route.load('d2first'), 5, state='path')
    assert recording.source == {'kind': 'state', 'name': 'start'}
    assert [state['kind'] for state in recording.states()] == ['end']
    assert _native.replay_verify(recording.data, states_dir=str(states_dir))['frames'] == recording.frames
    _, embedded = record(runner, Route.load('d2first'), 5)
    assert embedded.inputs().tobytes() == recording.inputs().tobytes() and len(embedded.data) > len(recording.data)
    # Another machine under the same name: the replay names the file it was recorded from and refuses this one.
    other = tmp_path / 'states'
    other.mkdir()
    shutil.copy(REPO_STATES_DIR / '0_77c_0.state', other / 'start.state')
    with pytest.raises(_native.ReplayError, match='start'):
        _native.replay_verify(recording.data, states_dir=str(other))
    with pytest.raises(_native.ReplayError):
        _native.replay_verify(recording.data, states_dir=str(tmp_path / 'nowhere'))


def test_eval_chain_records_the_scripted_segment_as_wins(scripted_models, tmp_path, capsys):
    folder, output = tmp_path / 'replays', tmp_path / 'chain.jsonl'
    chain = ['--models-dir', str(scripted_models), 'eval-chain', '--no-log', '--from', str(require_state(SEGMENT_STATE)),
             '--stop-at', f'step:{SEGMENT_STOP_STEP}']
    assert cli.main([*chain, '--runs', '2', '--parallel', '2', '--replays', str(folder), '--streams',
                     'decisions,rewards', '--output', str(output)]) == 0
    assert f'replays in {folder}' in capsys.readouterr().out
    assert sorted(p.name for p in folder.iterdir()) == ['00000000-win.tfreplay', '00000001-win.tfreplay']
    runs = [json.loads(line) for line in output.read_text().splitlines()]
    for path, run in zip(sorted(folder.iterdir()), runs):
        recording = _native.Replay(path.read_bytes())
        assert recording.meta['stop_at'] == f'step:{SEGMENT_STOP_STEP}' and recording.meta['seed'] == run['seed']
        assert recording.meta['ending'] == {'kind': 'win', 'result': 'success', 'outcome': 'milestone',
                                            'reason': run['reason']}
        assert len(recording.decisions()['frame']) == run['steps'] and recording.rewards()
        assert not [state for state in recording.states() if state['kind'] == 'keyframe']
        end = recording.states()[-1]
        assert (end['kind'], end['step']) == ('end', SEGMENT_STOP_STEP)
        assert _native.replay_verify(recording.data)['frames'] == recording.frames
    for bad, message in ((['--replays', str(folder), '--streams', 'sound'], "unknown replay stream(s) ['sound']"),
                         (['--streams', 'rewards'], '--streams needs --replays')):
        assert cli.main([*chain, '--runs', '1', '--output', str(tmp_path / 'bad.jsonl'), *bad]) == 2
        assert message in capsys.readouterr().err
    assert not (tmp_path / 'bad.jsonl').exists()


def test_a_power_on_replay_is_inputs_alone(scripted_models, states_dir):
    runner = Runner(models_dir=scripted_models, max_steps=12, states_dir=states_dir)
    result, recording = record(runner, Route.load('d2first', power_on={'delay': 7}), 3, streams=STREAMS)
    assert recording.source == {'kind': 'power-on', 'name': 'LINK', 'delay': 7}
    assert [state['kind'] for state in recording.states()][0] != 'start'
    inputs = recording.inputs()
    assert not inputs[:120].any() and inputs[120] == recording.meta['buttons']['START']
    assert not inputs[121:121 + 30 + 7].any() and inputs[121 + 30 + 7] == recording.meta['buttons']['START']
    control, room = recording.events()[:2]
    assert control['type'] == 'control' and control['frame'] > 121 + 30 + 7
    assert (room['type'], room['frame'], room['room'], room['step']) == ('room', control['frame'], 0x77, 0)
    edits = recording.edits()
    assert not (edits[:, 0] < control['frame']).any(), "nothing is written before the models take over"
    assert not ((edits[:, 1] >= 0x18) & (edits[:, 1] <= 0x24)).any(), "the game's random numbers are never written"
    assert len(recording.decisions()['frame']) == result.steps
    assert _native.replay_verify(recording.data)['frames'] == recording.frames
    # The same delay and seed: the same file, byte for byte.  Another delay: another game.
    assert record(runner, Route.load('d2first', power_on={'delay': 7}), 3, streams=STREAMS)[1].data == recording.data
    other = record(runner, Route.load('d2first', power_on={'delay': 8}), 3, streams=STREAMS)[1]
    assert other.end_ram()[0x18:0x25].tobytes() != recording.end_ram()[0x18:0x25].tobytes()


def test_a_replay_cut_at_a_keyframe_continues_and_stitches(chain_replays):
    folder, _, runner = chain_replays
    recording = _native.Replay(sorted(folder.iterdir())[0].read_bytes())
    states = recording.states()
    index = next(i for i, state in enumerate(states) if state['kind'] == 'keyframe')
    keyframe = states[index]
    cut = _native.Replay(_native.replay_truncate(recording.data, index))
    assert (cut.frames, cut.ending, cut.meta['ending']['result']) == (keyframe['frame'], 'unknown', 'truncated')
    assert cut.states()[-1]['state'] == keyframe['state'] and cut.inputs().tobytes() == recording.inputs()[
        :cut.frames].tobytes()
    assert _native.replay_verify(cut.data)['frames'] == cut.frames
    with pytest.raises(_native.ReplayError):
        _native.replay_truncate(recording.data, 0)          # the start state is no cut point

    onward = Route.load('d2first', checkpoint=keyframe)
    result, continued = record(runner, onward, 77, streams=STREAMS, keyframe_frames=60)
    assert continued.source == {'kind': 'checkpoint'} and continued.states()[0]['state'] == keyframe['state']
    assert result.rooms[0].room == '0x77' and result.steps <= 40

    joined = _native.Replay(_native.replay_stitch(cut.data, continued.data))
    assert joined.frames == cut.frames + continued.frames
    assert _native.replay_verify(joined.data)['frames'] == joined.frames
    assert np.array_equal(joined.end_ram(), continued.end_ram())
    kinds = [state['kind'] for state in joined.states()]
    assert (kinds[0], kinds[-1], kinds.count('checkpoint')) == ('start', 'end', 1)
    checkpoint = next(state for state in joined.states() if state['kind'] == 'checkpoint')
    assert (checkpoint['frame'], checkpoint['state']) == (cut.frames, keyframe['state'])
    assert [event['frame'] for event in joined.events() if event['type'] == 'segment'] == [cut.frames]
    assert len(joined.decisions()['frame']) == len(cut.decisions()['frame']) + len(continued.decisions()['frame'])
    # Two runs that don't meet don't join.
    with pytest.raises(_native.ReplayError):
        _native.replay_stitch(recording.data, continued.data)


def test_a_chain_started_on_the_segment_state_continues_from_its_win(scripted_models, states_dir):
    """The scripted segment's end state is a checkpoint on route step ``SEGMENT_STOP_STEP``."""
    runner = Runner(models_dir=scripted_models, max_steps=40, states_dir=states_dir)
    route = Route.load('d2first', start_from=str(require_state(SEGMENT_STATE)), stop_step=SEGMENT_STOP_STEP)
    result, recording = record(runner, route, 0)
    assert (result.result, recording.ending) == ('success', 'win')
    rooms = [event for event in recording.events() if event['type'] == 'room']
    assert [(room['room'], room['step']) for room in rooms] == [(0x3c, SEGMENT_START_STEP), (0x4c, SEGMENT_STOP_STEP)]
    assert recording.states()[-1]['step'] == SEGMENT_STOP_STEP


def _rng_edits(recording):
    """{address: value} of a recording's writes to the game's random numbers ($18-$24) before its first frame
    of play."""
    control = next(event['frame'] for event in recording.events() if event['type'] == 'control')
    return {int(addr): int(value) for frame, addr, _, value in recording.edits()
            if frame <= control and 0x18 <= addr <= 0x24}


def test_runs_that_start_from_a_state_differ_by_the_random_numbers_drawn_for_their_seed(scripted_models, states_dir):
    """What makes two argmax runs two games.  A start state's load redraws the game's random numbers from the run
    seed; a checkpoint is restored as it is unless asked (``randomize_rng``, the golden path's when its models
    don't all sample), and then the redraw is recorded as edits the stitched replay replays."""
    runner = Runner(models_dir=scripted_models, max_steps=40, states_dir=states_dir, deterministic=True)
    route = Route.load('d2first', start_from=str(require_state(SEGMENT_STATE)), stop_step=SEGMENT_STOP_STEP)
    (_, first), (_, again), (_, other) = (record(runner, route, seed) for seed in (0, 0, 1))
    assert first.data == again.data, "the same seed is the same run"
    assert len(_rng_edits(first)) == 13 and _rng_edits(first) != _rng_edits(other)
    assert all(model['deterministic'] for model in first.meta['models'])

    end = first.states()[-1]
    onward = Route.load('d2first', checkpoint=end)
    assert not record(runner, onward, 5)[1].edits()[:, 0].tolist().count(0), "a checkpoint as it is: no write"
    runner.max_steps = 6
    redrawn = Route.load('d2first', checkpoint={**end, 'randomize_rng': True})
    (_, five), (_, six) = (record(runner, redrawn, seed) for seed in (5, 6))
    assert len(_rng_edits(five)) == 13 and _rng_edits(five) != _rng_edits(six)
    assert five.states()[0]['state'] == end['state'], "the redraw is an edit, not another start state"
    joined = _native.Replay(_native.replay_stitch(first.data, five.data))
    assert _native.replay_verify(joined.data)['frames'] == first.frames + five.frames


def test_evaluation_episodes_are_recorded_with_their_endings(scripted_models, states_dir, tmp_path):
    checkpoint = registry.promoted_checkpoint('ow-start', scripted_models)
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=25)
    folder = tmp_path / 'episodes'
    result = evaluate_specialist(config, checkpoint, episodes_per_state=3, parallel=2, seed=4, states_dir=states_dir,
                                 replays=folder, streams=STREAMS, keyframe_frames=60)
    files = sorted(folder.iterdir())
    assert [path.name[:8] for path in files] == ['00000004', '00000005', '00000006']
    for path, episode in zip(files, result['records']):
        recording = _native.Replay(path.read_bytes())
        assert path.name == f"{episode['seed']:08d}-{recording.ending}.tfreplay"
        assert (recording.meta['kind'], recording.meta['specialist']) == ('eval', 'ow-start')
        ending, = [event for event in recording.events() if event['type'] == 'episode']
        assert (ending['ending'], ending['outcome'], ending['frame']) == (episode['ending'], episode['outcome'],
                                                                          recording.frames)
        assert len(recording.decisions()['frame']) == episode['steps'] and recording.rewards()
        assert _native.replay_verify(recording.data)['frames'] == recording.frames
        assert np.array_equal(end_of(recording), recording.end_ram())


def test_replay_folders_keep_the_wanted_endings(chain_replays, tmp_path):
    folder, _, _ = chain_replays
    data = sorted(folder.iterdir())[0].read_bytes()
    ending = _native.Replay(data).ending
    unwanted = next(name for name in replay.ENDINGS if name != ending)
    kept, dropped = replay.ReplayDir(tmp_path / 'kept', only=[ending]), replay.ReplayDir(tmp_path / 'dropped',
                                                                                         only=[unwanted])
    assert kept.add(12, data) == tmp_path / 'kept' / f'00000012-{ending}.tfreplay'
    assert dropped.add(12, data) is None and not list((tmp_path / 'dropped').iterdir())
    assert (kept.saved, dropped.saved, kept.endings, dropped.endings) == (1, 0, {ending: 1}, {ending: 1})
    assert (tmp_path / 'kept' / f'00000012-{ending}.tfreplay').read_bytes() == data
    with pytest.raises(ValueError, match='unknown ending'):
        replay.ReplayDir(tmp_path / 'x', only=['victory'])
    with pytest.raises(ValueError, match='unknown replay stream'):
        replay.record_options({}, ['sound'])
    assert replay.parse_streams('keyframes, rewards') == ('rewards', 'keyframes') and replay.parse_streams('') == ()
