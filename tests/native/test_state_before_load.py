"""`Env.state` before `reset()` and `Explorer.state` before `explore()` are RuntimeErrors, not Rust panics."""

import pytest

from triforce import _native as native


def test_env_state_needs_a_reset(states_dir):
    env = native.Env('ow-start', start_state='start', states_dir=str(states_dir))
    with pytest.raises(RuntimeError, match=r'call reset\(\) first'):
        _ = env.state
    env.reset(seed=0)
    assert env.state.location == 0x77


def test_explorer_state_needs_an_explore(states_dir):
    explorer = native.Explorer('ow-start', seed=3, states_dir=str(states_dir))
    with pytest.raises(RuntimeError, match=r'call explore\(\) first'):
        _ = explorer.state
