"""Ctrl-C stops long native calls within about a second (KeyboardInterrupt): an exploration, an Explorer search, a
rollout and Game.press. Each runs in a subprocess that is sent SIGINT 1.5 s into the call."""

import signal
import subprocess
import sys
import time

import pytest

from ..conftest import REPO_STATES_DIR

SETUP = """
import sys
from triforce import _native as native
states = sys.argv[1]
"""

# (setup, the long call)
CALLS = {
    'explore_states': ("", "native.explore_states('ow-start', sys.argv[2], states=['start'], iterations=10**6, "
                           "patience=10**6, parallel=2, states_dir=states)"),
    'explorer': ("explorer = native.Explorer('ow-start', seed=3, states_dir=states)",
                 "explorer.explore('start', iterations=10**6, patience=10**6)"),
    'rollout': ("""
from tests.networks import random_network
from triforce.rollout_buffer import RolloutBuffer
config = native.SpecialistConfig.load('ow-start')
pool = native.EnvPool(native.TrainTask(config, sys.argv[2], states_dir=states), num_envs=1, threads=1, seed=0)
views = RolloutBuffer(20000, 1, pool.obs_spec, 0.99, 0.95).rollout_buffers()
policy = random_network().to_native()
pool.wait_idle()
""", "pool.rollout([policy], 20000, views)"),
    'press': ("game = native.Game(states_dir=states)\ngame.load('start')", "game.press(0, 10**9)"),
}


@pytest.mark.parametrize('call', sorted(CALLS))
def test_ctrl_c_stops_a_long_native_call(call, tmp_path):
    setup, long_call = CALLS[call]
    script = f"{SETUP}{setup}\nprint('ready', flush=True)\n{long_call}\nprint('finished', flush=True)\n"
    process = subprocess.Popen([sys.executable, '-c', script, str(REPO_STATES_DIR), str(tmp_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == 'ready', process.stderr.read()[-2000:]
    time.sleep(1.5)                                  # inside the long call
    sent = time.monotonic()
    process.send_signal(signal.SIGINT)
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        pytest.fail(f'{call}: still running 5 s after SIGINT')
    assert time.monotonic() - sent < 5
    assert 'finished' not in stdout                  # it was interrupted, not done
    assert 'KeyboardInterrupt' in stderr, stderr[-2000:]
