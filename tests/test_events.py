"""Training event logs (``triforce.events``)."""

import json
from datetime import datetime

from triforce.events import EventLog, TrainingPlan, new_log_dir
from triforce.metrics import read_jsonl


def test_log_dirs_count_up_per_day(tmp_path):
    day1, day2 = datetime(2026, 11, 2, 9), datetime(2026, 11, 3, 9)
    names = [new_log_dir('r2c curriculum', tmp_path, day1).name, new_log_dir('x', tmp_path, day1).name,
             new_log_dir('x', tmp_path, day2).name]
    assert names == ['261102_001_r2c-curriculum', '261102_002_x', '261103_001_x']


def test_event_stream_continues_across_writers(tmp_path):
    """The training subprocess and the round driver append to the same stream one after the other."""
    path = tmp_path / 'ow-zora.jsonl'
    EventLog(path, specialist='ow-zora').emit('start', target_steps=10)
    EventLog(path, specialist='ow-zora').emit('end', status='completed')
    events = read_jsonl(str(path))
    assert [(e['seq'], e['event'], e['specialist']) for e in events] == [(0, 'start', 'ow-zora'),
                                                                          (1, 'end', 'ow-zora')]
    assert events[0]['target_steps'] == 10 and events[0]['ts'] <= events[1]['ts']


def test_plan_tracks_run_status_in_order(tmp_path):
    plan = TrainingPlan(tmp_path, 'r', 'round', [{'specialist': 'a'}, {'specialist': 'b'}])
    plan.update_run(0, status='running')
    running = json.loads((tmp_path / 'training.json').read_text())
    assert [r['status'] for r in running['runs']] == ['running', 'pending']
    assert running['runs'][0]['started'] and running['runs'][0]['finished'] is None

    plan.update_run(0, status='ok', steps=5)
    plan.finish('interrupted')
    done = json.loads((tmp_path / 'training.json').read_text())
    assert done['status'] == 'interrupted' and done['finished']
    assert done['runs'][0]['finished'] and done['runs'][0]['steps'] == 5 and done['runs'][1]['status'] == 'pending'
