"""``triforce.evaluate``: per-specialist evaluation over every seed start state, on the native eval pool."""

import json
import os
import subprocess
import sys

from triforce import _native
from triforce.evaluate import (_group, _linger_stats, _room_key, _visited_rooms, evaluate_specialist,
                               format_report)
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import observation_space
from triforce.registry import save_checkpoint

SMALL = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)


def test_evaluate_cli_uses_supplied_task_instead_of_reloading_its_name(states_dir, tmp_path):
    checkpoint = save_checkpoint('ow-start', 'alternate-task', ZeldaPolicy(observation_space(), SMALL), root=tmp_path)
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


def test_evaluate_specialist_covers_every_seed_state_reproducibly(states_dir, tmp_path):
    ckpt = save_checkpoint('ow-start', 'untrained', ZeldaPolicy(observation_space(), SMALL), root=tmp_path)
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=30)
    seen = []

    def evaluate(parallel=3, threads=1):
        return evaluate_specialist(config, ckpt, episodes_per_state=5, parallel=parallel, threads=threads, seed=5,
                                   states_dir=states_dir, on_episode=lambda r, done, total: seen.append((done, total)))

    result = evaluate()
    records = result['records']
    assert result['episodes'] == 5 == len(records)
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
    assert 'linger max/tail/worst' in format_report(result)
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
                {'dropped': 2, 'taken': 1, 'left': 1, 'wasted': 0.5}, blocks=4, bombs={'used': 3, 'cheated': 2}),
        _record((1, 0x44, False), '1_44w_1', False, 0.0, 1.5, 'death', blocks=0, bombs={'used': 1, 'cheated': 0}),
        _record((0, 0x77, True), '0_77cave_0', True, 3.0, 3.0, 'exit-correct', {'dropped': 1, 'taken': 1, 'left': 0}),
    ]
    rooms = _group(records, lambda r: _room_key(r['room']))
    assert list(rooms) == ['0:0x77:cave', '1:0x44']
    room = rooms['1:0x44']
    assert (room['episodes'], room['success_rate'], room['health_change']) == (2, 0.5, -1.25)
    assert room['health_change_total'] == (-1.0 + 0.5 - 1.5) / 2       # wasted hearts count as lost health
    assert (room['hearts_dropped'], room['hearts_taken'], room['hearts_left']) == (1.0, 0.5, 0.5)
    assert room['blocks'] == 2.0 and room['endings'] == {'exit-correct': 1, 'death': 1}
    assert (room['mean_bombs_used'], room['mean_bombs_cheated']) == (2.0, 1.0)
    assert 'blocks' not in rooms['0:0x77:cave']                         # never measured, not zero
    assert 'mean_bombs_used' not in rooms['0:0x77:cave']
    states = _group(records, lambda r: r['start_state'])
    assert list(states) == ['0_77cave_0', '1_44w_0', '1_44w_1']
    assert states['1_44w_1']['hearts_dropped'] == 0 and states['1_44w_1']['health_change_total'] == -1.5


def _visit(level, room, in_cave=False, *, steps=10, health_in=3.0, health_out=3.0, lost=0.0, gained=0.0, hits=0,
           dealt=0, drops=None, pickups=None, keys=0, exit_='n', bombs=0, cheated=0):
    return {'level': level, 'room': room, 'in_cave': in_cave, 'steps': steps, 'frames': steps * 8,
            'health_in': health_in, 'health_out': health_out, 'max_health': 3, 'health_lost': lost,
            'health_gained': gained, 'hits': hits, 'hits_dealt': dealt, 'drops': drops or {},
            'pickups': pickups or {}, 'keys_gained': keys, 'bombs_used': bombs, 'bombs_cheated': cheated,
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
        # Dodongo: two fights, hearts lost and bombs per fight.
        {'visits': [_visit(2, 0x0e, lost=1.0, bombs=3, cheated=1, exit_='success')]},
        {'visits': [_visit(2, 0x0e, lost=2.0, bombs=1, cheated=1, exit_='death')]},
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
    assert (dodongo['health_lost_per_visit'], dodongo['bombs_used_per_visit'], dodongo['bombs_cheated_per_visit']) \
        == (1.5, 2.0, 1.0)
    assert room['bombs_used_per_visit'] == 0.0
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
    assert lines[0].startswith('goriya: success 1/3 = 33%  health change -0.50 (total -0.75)')
    assert 'steps 42' in lines[0] and 'linger' not in lines[0]
    assert [line.split()[0] for line in lines[1:]] == ['1:0x45', '1:0x44']
    assert lines[1].split()[-1] == 'max-steps' and lines[2].split()[-1] == 'death'
