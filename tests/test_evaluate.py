"""``triforce.evaluate``: per-specialist evaluation over every seed start state, on the native eval pool."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import triforce
from triforce import _native, evaluate, provenance, registry, replay
from triforce.clips import episode_jobs
from triforce.evaluate import (_boss_stats, _clip_stats, _group, _linger_stats, _room_key, _start_health_fraction,
                               _visited_rooms,
                               default_output, eval_event, evaluate_specialist, format_report, read_evaluation,
                               write_result)
from triforce.model import ModelConfig
from triforce.registry import save_checkpoint

from .conftest import seeded_network

SMALL = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)


def test_evaluate_cli_uses_supplied_task_instead_of_reloading_its_name(states_dir, tmp_path):
    checkpoint = save_checkpoint('ow-start', 'alternate-task', seeded_network(1, SMALL), root=tmp_path)
    config = tmp_path / 'alternate.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 1\n")
    output = tmp_path / 'eval.json'
    subprocess.run([sys.executable, '-m', 'triforce.evaluate', '--specialist', str(config),
                    '--checkpoint', str(checkpoint), '--episodes-per-state', '1', '--output', str(output)],
                   check=True, capture_output=True, text=True,
                   env={**os.environ, 'TRIFORCE_STATES_DIR': str(states_dir)})
    result = json.loads(output.read_text())
    assert result['episodes'] == 1
    assert result['endings'] == {'max-steps': 1}
    assert result['records'][0]['steps'] == 1


def test_evaluate_cli_logs_a_folder_the_training_gui_reads(states_dir, tmp_path):
    checkpoint = save_checkpoint('ow-start', 'logged', seeded_network(2, SMALL), root=tmp_path)
    config = tmp_path / 'short.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 3\n")
    logs = tmp_path / 'logs'
    assert evaluate.main(['--specialist', str(config), '--checkpoint', str(checkpoint), '--episodes-per-state', '3',
                          '--parallel', '2', '--output', str(tmp_path / 'eval.json'), '--logs-dir', str(logs)]) == 0
    (folder,) = (p for p in logs.iterdir() if p.is_dir())       # beside new_log_dir's .lock
    plan = json.loads((folder / 'training.json').read_text())
    assert (plan['kind'], plan['status'], plan['runs'][0]['status']) == ('eval', 'completed', 'ok')
    events = [json.loads(line) for line in (folder / 'ow-start.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events][0] == 'eval_start' and [e['event'] for e in events][-2:] == ['eval', 'end']
    assert events[0]['episodes'] == 3 and events[-2]['episodes'] == 3 and events[-1]['status'] == 'ok'
    assert evaluate.main(['--specialist', str(config), '--checkpoint', str(checkpoint), '--episodes-per-state', '1',
                          '--output', str(tmp_path / 'eval.json'), '--logs-dir', str(logs), '--no-log']) == 0
    assert len([p for p in logs.iterdir() if p.is_dir()]) == 1


def _short_config(tmp_path, name='ow-start'):
    config = tmp_path / f'{name}.yaml'
    config.write_text(f"name: {name}\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 1\n")
    return config


def test_only_the_canonical_evaluation_is_written_to_eval_json(states_dir, tmp_path):
    """F125: another act mode, episode count, seed or specialist goes to its own file; eval.json is untouched."""
    checkpoint = save_checkpoint('ow-start', 'canon', seeded_network(3, SMALL), root=tmp_path)
    run = checkpoint.parent
    config = _short_config(tmp_path)

    def main(*extra):
        return evaluate.main(['--specialist', str(config), '--checkpoint', str(run), '--no-log', *extra])

    assert main() == 0
    canonical = (run / 'eval.json').read_text()
    assert json.loads(canonical)['episodes'] == 100
    assert main('--act', 'argmax', '--episodes-per-state', '1') == 0
    assert main('--act', 'sample', '--seed', '2', '--episodes-per-state', '1') == 0
    assert main('--specialist', str(_short_config(tmp_path, 'ow-zora')), '--episodes-per-state', '1') == 0
    assert (run / 'eval.json').read_text() == canonical
    assert sorted(p.name for p in run.glob('eval*.json')) == [
        'eval-argmax-e1.json', 'eval-e1-s2.json', 'eval-ow-zora-e1.json', 'eval.json']
    assert json.loads((run / 'eval-argmax-e1.json').read_text())['act'] == 'argmax'
    # --act argmax on a model that acts by argmax is its canonical evaluation
    registry.set_act_mode(run, 'argmax')
    assert default_output(run, 'ow-start', 'argmax', 100, 0) == run / 'eval.json'
    assert default_output(run, 'ow-start', 'sample', 100, 0) == run / 'eval-sample.json'
    assert default_output(run, 'ow-start', 'model', 100, 0) == run / 'eval.json'


def test_write_result_keeps_the_old_file_when_the_write_fails(tmp_path, monkeypatch):
    """F125/F126: a failed write leaves the previous eval.json whole."""
    path = write_result({'episodes': 3}, tmp_path / 'run' / 'eval.json')

    def fail(*_args, **_kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(json, 'dump', fail)
    with pytest.raises(OSError):
        write_result({'episodes': 1}, path)
    assert json.loads(path.read_text()) == {'episodes': 3}
    assert [p.name for p in path.parent.iterdir()] == ['eval.json']


def test_read_evaluation_finds_the_evaluation_in_the_act_mode_of_the_current_weights(tmp_path):
    """F027: a reader gets the evaluation matching the run's act mode and checkpoint, or a LookupError."""
    checkpoint = save_checkpoint('ow-start', 'r', seeded_network(4, SMALL), root=tmp_path)
    run = checkpoint.parent
    sha1 = registry.checkpoint_sha1(run)
    write_result({'deterministic': False, 'success_rate': 0.5}, run / 'eval.json')       # an old file, no sha1
    assert read_evaluation(run) == (run / 'eval.json', {'deterministic': False, 'success_rate': 0.5})
    with pytest.raises(LookupError, match='eval.json was measured by sample'):
        read_evaluation(run, 'argmax')
    registry.set_act_mode(run, 'argmax')      # renames the sampled eval.json
    with pytest.raises(LookupError, match='no eval.json or eval-argmax.json'):
        read_evaluation(run)
    assert read_evaluation(run, 'sample')[0] == run / 'eval-sample.json'
    write_result({'act': 'argmax', 'deterministic': True, 'checkpoint_sha1': 'f' * 40}, run / 'eval-argmax.json')
    with pytest.raises(LookupError, match='other weights'):
        read_evaluation(run)
    write_result({'act': 'argmax', 'deterministic': True, 'checkpoint_sha1': sha1}, run / 'eval-argmax.json')
    assert read_evaluation(run)[0] == run / 'eval-argmax.json'


def test_start_health_fraction_uses_the_max_hearts_at_the_start():
    """F095: a heart container taken during the episode doesn't shrink the start fraction."""
    record = {'start_health': 2.0, 'start_max_health': 3, 'health': 3.0, 'max_health': 4}
    assert _start_health_fraction([record]) == pytest.approx(2 / 3)
    old = {'start_health': 2.0, 'health': 3.0, 'max_health': 4}     # records written before start_max_health
    assert _start_health_fraction([old]) == pytest.approx(0.5)


def test_evaluation_names_its_inputs(states_dir, tmp_path):
    """F080 (2): the result carries the act mode, the checkpoint's sha1, the network hash and the config."""
    checkpoint = save_checkpoint('ow-start', 'p', seeded_network(5, SMALL), root=tmp_path)
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=3)
    result = evaluate_specialist(config, checkpoint, episodes_per_state=1, states_dir=states_dir,
                                 deterministic=True)
    assert (result['act'], result['deterministic']) == ('argmax', True)
    assert result['checkpoint_sha1'] == registry.checkpoint_sha1(checkpoint)
    assert result['model_hash'] == f"{registry.load_model(checkpoint)[0].to_native().hash:016x}"
    assert result['code'] == provenance.code_version() and result['data'] == provenance.data_hash()
    assert result['version'] == triforce.__version__
    again = _native.SpecialistConfig.from_json(json.dumps(result['config']))
    assert (again.name, again.max_steps, again.start_states) == ('ow-start', 3, ['start'])
    assert json.loads(json.dumps(result)) == result


def test_evaluation_acts_the_way_the_run_is_registered_unless_told(states_dir, tmp_path):
    """F129: ``deterministic=None`` (``--act model``) follows the run's act file; True/False and ``--act`` override
    it. The recordings name the mode and their inputs are the ones that mode plays."""
    checkpoint = save_checkpoint('ow-start', 'act', seeded_network(7, SMALL), root=tmp_path)
    config = tmp_path / 'ow-start.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 20\n")

    def played(result, folder):
        """(result's act, the recordings' deterministic flag, every episode's inputs by seed)."""
        files = sorted(folder.iterdir())
        assert [f.name[:8] for f in files] == ['00000000', '00000001'] and all(f.suffix == '.tfreplay' for f in files)
        recordings = [_native.Replay(f.read_bytes()) for f in files]
        flag, = {r.meta['models'][0]['deterministic'] for r in recordings}
        assert all(r.meta['kind'] == 'eval' and r.meta['specialist'] == 'ow-start' for r in recordings)
        return result['act'], flag, [r.inputs().tobytes() for r in recordings]

    def play(name, deterministic):
        folder = tmp_path / name
        return played(evaluate_specialist(str(config), checkpoint, episodes_per_state=2, states_dir=states_dir,
                                          deterministic=deterministic, replays=folder), folder)

    def cli(name, act):
        folder, output = tmp_path / name, tmp_path / f'{name}.json'
        assert evaluate.main(['--specialist', str(config), '--checkpoint', str(checkpoint), '--no-log', '--act', act,
                              '--episodes-per-state', '2', '--replays', str(folder), '--output', str(output)]) == 0
        return played(json.loads(output.read_text()), folder)

    sampled, argmax = play('sampled', False), play('argmax', True)
    assert sampled[:2] == ('sample', False) and argmax[:2] == ('argmax', True)
    assert sampled[2] != argmax[2], "argmax and sampling play differently"
    assert play('unregistered', None) == sampled == cli('model-sample', 'model')
    registry.set_act_mode(checkpoint.parent, 'argmax')
    assert play('registered', None) == argmax == cli('model-argmax', 'model')
    assert cli('sample', 'sample') == sampled
    with pytest.raises(SystemExit):
        evaluate.main(['--specialist', str(config), '--checkpoint', str(checkpoint), '--streams', 'rewards'])


def test_evaluate_specialist_covers_every_seed_state_reproducibly(states_dir, tmp_path):
    ckpt = save_checkpoint('ow-start', 'untrained', seeded_network(6, SMALL), root=tmp_path)
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=30)
    seen = []

    def evaluate(parallel=3, threads=1):
        return evaluate_specialist(config, ckpt, episodes_per_state=5, parallel=parallel, threads=threads, seed=5,
                                   states_dir=states_dir, on_episode=lambda r, done, total: seen.append((done, total)),
                                   clips='on')

    result = evaluate()
    records = result['records']
    assert result['episodes'] == 5 == len(records) and result['clips'] == 'on'
    assert all(r['clips_allowed'] and r['clips_fixed'] is None for r in records)
    assert set(result['by_clips']) == {'on'} and result['by_clips']['on']['success_rate'] == result['success_rate']
    assert seen == [(i, 5) for i in range(1, 6)]
    assert set(result['by_state']) == {'start'} and result['by_state']['start']['episodes'] == 5
    assert [r['seed'] for r in records] == list(range(5, 10))
    assert all(r['outcome'] in {'success', 'hard-failure', 'soft-failure'} for r in records)
    assert all(0 < r['steps'] <= 30 for r in records)
    assert result['success_rate'] == sum(r['success'] for r in records) / 5
    assert set(result['by_room']) == {'0:0x77'}
    assert all(set(r['hearts']) == {'dropped', 'taken', 'healed', 'wasted', 'left', 'expired'} for r in records)
    assert result['hearts_dropped'] == sum(r['hearts']['dropped'] for r in records) / 5
    assert result['mean_health_change_total'] == result['health_change_total'] >= result['mean_health_change']
    assert {'health_change_total', 'hearts_dropped', 'hearts_taken', 'hearts_left'} <= set(result['by_room']['0:0x77'])
    assert all({'max_steps', 'max_frames', 'tail_steps', 'tail_frames', 'steps', 'frames',
                'boss_kill_steps', 'boss_kill_frames', 'heart_container_steps', 'heart_container_frames'}
               == set(r['linger']) for r in records)
    assert result['linger_max_steps'] == sum(r['linger']['max_steps'] for r in records) / 5
    assert result['linger_worst_steps'] == max(r['linger']['max_steps'] for r in records)
    assert result['by_room']['0:0x77']['linger_max_steps'] == result['linger_max_steps']
    assert all('linger_max_steps' in stats for stats in result['by_state'].values())
    assert 'linger' in format_report(result)
    # Room visits (format 2): they cover each episode and visited_rooms counts every one.
    assert result['format'] == 2
    assert all(r['visits'] and sum(v['steps'] for v in r['visits']) == r['steps'] for r in records)
    assert all((v['level'], v['room'], v['in_cave']) == tuple(r['room']) for r in records for v in r['visits'][:1])
    assert all(r['visits'][-1]['health_out'] == r['health'] for r in records)
    assert sum(stats['visits'] for stats in result['visited_rooms'].values()) == sum(len(r['visits'])
                                                                                     for r in records)
    assert result['visited_rooms']['0:0x77']['episodes'] == 5
    assert result['visited_rooms']['0:0x77']['specialist'] == 'ow-start'

    # Sampled actions are keyed to (episode seed, decision): the same records whichever env or thread plays them.
    assert evaluate()['records'] == records
    assert evaluate(parallel=3, threads=3)['records'] == records
    assert evaluate(parallel=2, threads=2)['records'] == records


def test_a_clip_family_is_evaluated_with_the_switch_on_and_off(states_dir, tmp_path):
    """Round 22: ``--clips both`` (a clip family's default) plays every start state with the run-time clips switch
    on, then off, seeds following on; the switch-on half is the ``on`` evaluation.  The top-level figures are over
    both, ``by_clips`` holds each; a family without clips plays ``on`` only."""
    ckpt = save_checkpoint('ow-start', 'untrained', seeded_network(6, SMALL), root=tmp_path)
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=20)
    run = lambda clips: evaluate_specialist(config, ckpt, episodes_per_state=3, parallel=3, seed=5,
                                            states_dir=states_dir, clips=clips)
    both, on, off = run(None), run('on'), run('off')
    assert both['clips'] == 'both' and both['episodes'] == 6
    assert [(r['seed'], r['clips_allowed']) for r in both['records']] == [(5, True), (6, True), (7, True),
                                                                          (8, False), (9, False), (10, False)]
    assert both['records'][:3] == on['records']
    assert [r['clips_allowed'] for r in off['records']] == [False] * 3
    assert both['success_rate'] == sum(r['success'] for r in both['records']) / 6
    assert set(both['by_clips']) == {'on', 'off'}
    assert both['by_clips']['on']['episodes'] == both['by_clips']['off']['episodes'] == 3
    assert both['by_clips']['on']['success_rate'] == on['success_rate']
    assert set(both['by_clips']['off']['by_room']) == {'0:0x77'} and 'visited_rooms' in both['by_clips']['off']
    event = eval_event(both)
    assert event['clips'] == 'both' and set(event['by_clips']) == {'on', 'off'}
    assert 'clips on ' in format_report(both) and 'clips off' in format_report(both)
    plain = evaluate_specialist(config.replace(clips=False), ckpt, episodes_per_state=2, parallel=2, seed=5,
                                states_dir=states_dir)
    assert plain['clips'] == 'on' and plain['episodes'] == 2


def test_the_route_families_play_each_start_state_in_the_switch_states_it_allows(states_dir):
    """Round 22's route: ``--clips both`` plays a start state in both switch states where both ways of a stretch
    enter its room that way (or neither: a room off the stretches), and only in its way's switch state where one way
    alone does (``triforce.clips.episode_jobs``, from the installed states)."""
    del states_dir                                                           # requires the ROM
    installed = Path(triforce.__file__).parent / 'states'
    expect = {
        'ow-octorok': {'0_67c': 'both', '0_67e': 'both', '0_68n': 'both', '0_66e': 'off', '0_64e': 'off',
                       '0_64w': 'both'},
        'ow-leever': {'0_75e': 'on', '0_48n': 'both'},
        'ow-tektite': {'0_76e': 'on', '0_74e': 'on', '0_74w': 'off', '0_74c': 'both'},
        'ow-start': {'0_77n': 'on'},
        'ow-zora': {'0_65e': 'off', '0_65w': 'both'},
        'ow-moblin': {'0_63e': 'off', '0_73n': 'off', '0_63s': 'both', '0_73e': 'both'},
        'vire': {'4_31e': 'both', '4_30e': 'off', '4_20s': 'off', '4_31w': 'both', '4_12n': 'both'},
        'stalfos-keese-gel': {'4_21s': 'on', '4_21w': 'off', '4_01s': 'both', '4_51s': 'both'},
    }
    for name, rooms in expect.items():
        jobs = episode_jobs(_native.SpecialistConfig.load(name), 1, mode='both', states_dir=installed)
        played = {}
        for state, _, on in jobs:
            played.setdefault(state, set()).add(on)
        for prefix, mode in rooms.items():
            chains = [state for state in played if state.startswith(prefix + '_')]
            assert len(chains) == 3, (name, prefix, sorted(played))
            want = {'both': {True, False}, 'on': {True}, 'off': {False}}[mode]
            assert all(played[state] == want for state in chains), (name, prefix, {s: played[s] for s in chains})


def test_linger_evaluation_aggregates_only_measured_records():
    base = {'health': 2, 'start_health': 3, 'success': True, 'ending': 'exit-correct'}
    records = [{**base, 'start_state': 'a', 'room': 'room-a'},
               {**base, 'start_state': 'a', 'room': 'room-a',
                'linger': {'max_steps': 10, 'max_frames': 200, 'tail_steps': 0, 'tail_frames': 0}},
               {**base, 'start_state': 'b', 'room': 'room-a',
                'linger': {'max_steps': 30, 'max_frames': 100, 'tail_steps': 10, 'tail_frames': 50}},
               {**base, 'start_state': 'c', 'room': 'room-b'}]
    assert _linger_stats(records) == {'linger_max_steps': 20, 'linger_max_frames': 150,
                                     'linger_tail_steps': 5, 'linger_tail_frames': 25,
                                     'linger_worst_steps': 30, 'linger_worst_frames': 200}
    rooms = _group(records, lambda r: r['room'])
    states = _group(records, lambda r: r['start_state'])
    assert rooms['room-a']['linger_max_steps'] == 20
    assert rooms['room-a']['linger_worst_frames'] == 200
    assert states['a']['linger_max_steps'] == 10
    assert states['a']['linger_tail_steps'] == 0
    assert states['b']['linger_max_steps'] == 30
    assert not any(key.startswith('linger_') for key in rooms['room-b'])
    assert not any(key.startswith('linger_') for key in states['c'])
    assert _linger_stats([]) == _linger_stats([base]) == {}


def _record(room, state, success, health, start_health, ending, hearts=None, **extra):
    return {'room': room, 'start_state': state, 'success': success, 'health': health, 'start_health': start_health,
            'ending': ending, 'hearts': hearts, **extra}


def test_room_and_state_groups_average_health_hearts_blocks_and_count_endings():
    records = [
        _record((1, 0x44, False), '1_44w_0', True, 2.0, 3.0, 'exit-correct',
                {'dropped': 2, 'taken': 1, 'left': 1, 'wasted': 0.5}, blocks=4, bombs={'used': 3},
                cheated={'bombs': 2, 'keys': 1, 'rupees': 0}),
        _record((1, 0x44, False), '1_44w_1', False, 0.0, 1.5, 'death', blocks=0, bombs={'used': 1},
                cheated={'bombs': 0, 'keys': 0, 'rupees': 0}),
        _record((0, 0x77, True), '0_77cave_0', True, 3.0, 3.0, 'exit-correct', {'dropped': 1, 'taken': 1, 'left': 0}),
    ]
    rooms = _group(records, lambda r: _room_key(r['room']))
    assert list(rooms) == ['0:0x77:cave', '1:0x44']
    room = rooms['1:0x44']
    assert (room['episodes'], room['success_rate'], room['health_change']) == (2, 0.5, -1.25)
    assert room['health_change_total'] == (-1.0 + 0.5 - 1.5) / 2       # wasted hearts count as lost health
    assert (room['hearts_dropped'], room['hearts_taken'], room['hearts_left']) == (1.0, 0.5, 0.5)
    assert room['blocks'] == 2.0 and room['endings'] == {'exit-correct': 1, 'death': 1}
    assert room['mean_bombs_used'] == 2.0
    assert (room['mean_cheated_bombs'], room['mean_cheated_keys'], room['mean_cheated_rupees']) == (1.0, 0.5, 0.0)
    assert 'blocks' not in rooms['0:0x77:cave']                         # never measured, not zero
    assert 'mean_bombs_used' not in rooms['0:0x77:cave'] and 'mean_cheated_keys' not in rooms['0:0x77:cave']
    states = _group(records, lambda r: r['start_state'])
    assert list(states) == ['0_77cave_0', '1_44w_0', '1_44w_1']
    assert states['1_44w_1']['hearts_dropped'] == 0 and states['1_44w_1']['health_change_total'] == -1.5


def test_clip_means_and_the_state_target_grouping():
    def clips(done, aborted, moved=0):
        return {'done': done, 'aborted': aborted, 'done_n': 0, 'done_s': 0, 'done_w': done, 'done_e': 0,
                'finished_by_move': moved}

    records = [
        _record((0, 0x59, False), '0_59w_0', True, 3.0, 3.0, 'exit-correct', clips=clips(1, 0), target='S'),
        _record((0, 0x59, False), '0_59w_0', False, 3.0, 3.0, 'exit-wrong', clips=clips(0, 1, 1), target='E'),
        _record((0, 0x59, False), '0_59w_0', True, 3.0, 3.0, 'exit-correct', clips=clips(3, 1), target='E'),
        _record((0, 0x59, False), '0_59w_1', True, 3.0, 3.0, 'exit-correct'),          # older: no counters
    ]
    assert _clip_stats(records) == {'mean_clips_done': 4 / 3, 'mean_clips_aborted': 2 / 3, 'mean_clips_done_n': 0,
                                    'mean_clips_done_s': 0, 'mean_clips_done_w': 4 / 3, 'mean_clips_done_e': 0,
                                    'mean_clips_finished_by_move': 1 / 3}
    groups = _group(records, lambda r: f"{r['start_state']}:{r.get('target', '-')}")
    assert list(groups) == ['0_59w_0:E', '0_59w_0:S', '0_59w_1:-']
    assert (groups['0_59w_0:E']['success_rate'], groups['0_59w_0:E']['mean_clips_done']) == (0.5, 1.5)
    assert groups['0_59w_0:S']['mean_clips_aborted'] == 0 and 'mean_clips_done' not in groups['0_59w_1:-']


def test_boss_stats_tell_stun_kills_from_swallow_kills():
    def fight(success, stuns, eats):
        return {'success': success, 'boss': {'stuns': stuns, 'eats': eats}}

    records = [
        fight(True, 1, 0),      # stun and sword
        fight(True, 2, 0),      # stun and sword, after a stun that wore off
        fight(True, 0, 2),      # two swallowed bombs
        fight(True, 1, 1),      # mixed: not a clean stun kill
        fight(False, 1, 0),     # failures don't count toward the rate
        {'success': True},      # an older record without the field: not measured
    ]
    stats = _boss_stats(records)
    assert (stats['mean_boss_stuns'], stats['mean_boss_eats']) == (1.0, 0.6)
    assert stats['stun_kill_rate'] == 0.5
    assert _boss_stats([fight(False, 0, 1)]) == {'mean_boss_stuns': 0.0, 'mean_boss_eats': 1.0}
    assert _boss_stats([{'success': True}]) == {}
    rooms = _group([{**r, 'room': (2, 0x0e, False), 'start_state': '2_0es_0', 'health': 3.0, 'start_health': 3.0,
                     'ending': 'success'} for r in records], lambda r: _room_key(r['room']))
    assert rooms['2:0x0e']['stun_kill_rate'] == 0.5 and rooms['2:0x0e']['mean_boss_eats'] == 0.6


def _visit(level, room, in_cave=False, *, steps=10, health_in=3.0, health_out=3.0, lost=0.0, gained=0.0, hits=0,
           dealt=0, drops=None, pickups=None, keys=0, exit_='n', bombs=0, cheated_bombs=0, cheated_keys=0,
           stuns=0, eats=0):
    return {'level': level, 'room': room, 'in_cave': in_cave, 'steps': steps, 'frames': steps * 8,
            'health_in': health_in, 'health_out': health_out, 'max_health': 3, 'health_lost': lost,
            'health_gained': gained, 'hits': hits, 'hits_dealt': dealt, 'drops': drops or {},
            'pickups': pickups or {}, 'keys_gained': keys, 'bombs_used': bombs, 'cheated_bombs': cheated_bombs,
            'cheated_keys': cheated_keys, 'cheated_rupees': 0, 'boss_stuns': stuns, 'boss_eats': eats,
            'exit': exit_}


def test_visited_rooms_average_every_visit_of_every_room():
    records = [
        {'visits': [_visit(1, 0x73, steps=20, health_out=2.0, lost=1.5, gained=0.5, hits=2, dealt=3,
                           drops={'heart': 1, 'rupee': 2}, pickups={'heart': 1}, keys=1, exit_='n'),
                    _visit(1, 0x63, health_in=2.0, health_out=0.0, lost=2.0, hits=1, exit_='death')]},
        {'visits': [_visit(1, 0x73, steps=10, dealt=1, drops={'bombs': 1}, exit_='w'),
                    _visit(1, 0x72, exit_='e'),
                    _visit(1, 0x73, steps=30, exit_='timeout')]},
        {'visits': [_visit(0, 0x77, True, exit_='cave')]},
        # Dodongo: two fights, hearts lost, bombs, stuns and swallows per fight.
        {'visits': [_visit(2, 0x0e, lost=1.0, bombs=3, cheated_bombs=1, stuns=1, eats=0, exit_='success')]},
        {'visits': [_visit(2, 0x0e, lost=2.0, bombs=1, cheated_bombs=1, cheated_keys=1, stuns=0, eats=1,
                           exit_='death')]},
        {},                                                                 # a format-1 record: no visits
    ]
    rooms = _visited_rooms(records, 'goriya')
    assert list(rooms) == ['0:0x77:cave', '1:0x63', '1:0x72', '1:0x73', '2:0x0e']
    room = rooms['1:0x73']
    assert (room['level'], room['room'], room['in_cave'], room['specialist']) == (1, 0x73, False, 'goriya')
    assert (room['visits'], room['episodes'], room['deaths'], room['death_rate']) == (3, 2, 0, 0.0)
    assert room['steps_per_visit'] == 20.0
    assert (room['health_lost_per_visit'], room['health_gained_per_visit']) == (0.5, 0.5 / 3)
    assert room['health_change_per_visit'] == -1 / 3
    assert (room['hits_per_visit'], room['no_hit_rate'], room['hits_dealt_per_visit']) == (2 / 3, 2 / 3, 4 / 3)
    assert room['keys_per_visit'] == 1 / 3
    assert room['drops_per_visit'] == {'bombs': 1 / 3, 'rupee': 2 / 3, 'heart': 1 / 3}
    assert room['pickups_per_visit'] == {'heart': 1 / 3}
    assert room['exits'] == {'n': 1, 'w': 1, 'timeout': 1}
    died = rooms['1:0x63']
    assert (died['deaths'], died['death_rate'], died['health_change_per_visit']) == (1, 1.0, -2.0)
    assert died['drops_per_visit'] == {} and died['no_hit_rate'] == 0.0
    assert rooms['0:0x77:cave']['in_cave'] and rooms['0:0x77:cave']['exits'] == {'cave': 1}
    dodongo = rooms['2:0x0e']
    assert (dodongo['health_lost_per_visit'], dodongo['bombs_used_per_visit']) == (1.5, 2.0)
    assert (dodongo['cheated_bombs_per_visit'], dodongo['cheated_keys_per_visit'],
            dodongo['cheated_rupees_per_visit']) == (1.0, 0.5, 0.0)
    assert (dodongo['boss_stuns_per_visit'], dodongo['boss_eats_per_visit']) == (0.5, 0.5)
    assert room['bombs_used_per_visit'] == 0.0 and room['boss_stuns_per_visit'] == 0.0
    assert room['cheated_keys_per_visit'] == 0.0 and 'bombs_cheated_per_visit' not in room
    assert _visited_rooms([{}], 'goriya') == {}


def test_report_lists_rooms_worst_first_with_their_top_ending():
    result = {'specialist': 'goriya', 'episodes': 3, 'success_rate': 1 / 3, 'mean_health_change': -0.5,
              'mean_health_change_total': -0.75, 'hearts_dropped': 1.0, 'hearts_taken': 0.5, 'hearts_left': 0.5,
              'mean_steps': 41.6,
              'records': [{'success': True}, {'success': False}, {'success': False}],
              'by_room': {
                  '1:0x44': {'success_rate': 0.5, 'health_change': -0.25, 'health_change_total': -0.5,
                             'hearts_dropped': 1, 'hearts_taken': 0, 'hearts_left': 1, 'episodes': 2,
                             'endings': {'exit-correct': 1, 'death': 2}},
                  '1:0x45': {'success_rate': 0.0, 'health_change': -1.0, 'health_change_total': -1.0,
                             'hearts_dropped': 0, 'hearts_taken': 0, 'hearts_left': 0, 'episodes': 1,
                             'endings': {'max-steps': 1}}}}
    lines = format_report(result).splitlines()
    numbers = [float(n) for n in re.findall(r'[-+]?\d+(?:\.\d+)?', lines[0])]
    assert lines[0].split(':')[0] == 'goriya' and numbers[:5] == [1, 3, 33, -0.5, -0.75]   # 1/3 = 33%, -0.50 (-0.75)
    assert 'steps 42' in lines[0] and 'linger' not in lines[0]
    assert [line.split()[0] for line in lines[1:]] == ['1:0x45', '1:0x44']
    assert lines[1].split()[-1] == 'max-steps' and lines[2].split()[-1] == 'death'
