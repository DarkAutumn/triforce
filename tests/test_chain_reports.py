"""The hand-analysis reports scripts/heart_stats.py and scripts/chain_rooms.py on eval and chain records."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


heart_stats = _script('heart_stats')
chain_rooms = _script('chain_rooms')


def _visit(index, room, outcome, specialist='ow-tektite', steps=10, health_in=3.0, health_out=2.0):
    return {'index': index, 'level': 0, 'room': room, 'in_cave': False, 'specialist': specialist, 'steps': steps,
            'frames': steps * 8, 'health_in': health_in, 'health_out': health_out, 'max_health': 3,
            'outcome': outcome, 'hearts_dropped': 0, 'hearts_taken': 0, 'hearts_healed': 0.0,
            'hearts_wasted': 0.0, 'hearts_left': 0, 'linger': {'max_steps': steps}}


def _chain_file(tmp_path):
    """One successful run that crosses 0:0x74 and ends with the milestone marker in the same room."""
    run = {'route': 'd2first', 'seed': 0, 'result': 'success', 'reason': '', 'steps': 30, 'frames': 240,
           'retreats': 0, 'rooms': [_visit(0, '0x73', 'exit-correct'), _visit(1, '0x74', 'exit-correct', steps=20),
                                    _visit(2, '0x74', 'milestone', specialist='-', steps=0, health_in=2.0)]}
    path = tmp_path / 'chain.jsonl'
    path.write_text(json.dumps(run) + '\n')
    return path


def test_heart_stats_groups_eval_records_by_formatted_room(tmp_path, capsys):
    hearts = {'dropped': 1, 'taken': 1, 'healed': 0.5, 'wasted': 0.0, 'left': 0}
    record = {'start_state': '1_23s_0', 'room': [1, 0x23, False], 'seed': 0, 'success': True, 'health': 2.5,
              'max_health': 3, 'start_health': 3.0, 'hearts': hearts}
    path = tmp_path / 'eval.json'
    path.write_text(json.dumps({'format': 2, 'specialist': 'goriya', 'records': [record, record]}))
    heart_stats.summarize([path], by_room=True)
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    assert [row[:3] for row in rows] == [['goriya', '1:0x23', '2']]


def test_heart_stats_skips_the_milestone_marker(tmp_path, capsys):
    heart_stats.summarize([_chain_file(tmp_path)], by_room=True)
    groups = [line.split()[:3] for line in capsys.readouterr().out.splitlines()[2:]]
    assert groups == [['ow-tektite', '0:0x73', '1'], ['ow-tektite', '0:0x74', '1']]


def test_chain_rooms_counts_only_played_visits(tmp_path, capsys):
    chain_rooms.main([str(_chain_file(tmp_path))])
    rows = {line.split()[0]: line.split() for line in capsys.readouterr().out.splitlines()[2:]}
    assert rows['0:0x74'][1:2] == ['1'] and rows['0:0x74'][4] == '20.0'
