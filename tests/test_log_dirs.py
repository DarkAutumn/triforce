"""Log folders of standalone jobs: ``events.new_log_dir`` numbering and ``events.logged_job`` endings."""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

from triforce import events

DAY = datetime(2026, 11, 10)


def test_numbers_keep_counting_past_999(tmp_path):
    (tmp_path / '261110_999_x').mkdir()
    (tmp_path / '261110_notes').mkdir()                 # not a numbered folder: ignored
    first = events.new_log_dir('y', tmp_path, now=DAY)
    second = events.new_log_dir('y', tmp_path, now=DAY)
    third = events.new_log_dir('z', tmp_path, now=DAY)
    assert [p.name for p in (first, second, third)] == ['261110_1000_y', '261110_1001_y', '261110_1002_z']


JOB = '''
import sys, time
from triforce import events
with events.logged_job('sleepy', 'chain', [], sys.argv[1]):
    print('ready', flush=True)
    time.sleep(60)
'''


def test_sigterm_ends_a_logged_job_interrupted(tmp_path):
    with subprocess.Popen([sys.executable, '-c', JOB, str(tmp_path)], stdout=subprocess.PIPE, text=True,
                          env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}) as process:
        assert process.stdout.readline().strip() == 'ready'
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) != 0 and time.monotonic() - started < 10
    folder, = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert json.loads((folder / events.PLAN_FILE).read_text())['status'] == 'interrupted'
