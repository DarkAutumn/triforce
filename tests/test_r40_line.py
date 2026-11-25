"""Round 40: 3:0x69 crossed by the owner's line (``scripts/experiments/r40_d3_69_line.py``; decisions 2026-11-23
"3:0x69 is crossed by a line").  The rule's walks on fixed seeds through the darknut task's environment: the way out
along the line (no Darknut struck), and the way in where a Darknut stands in the line's column and is struck and
walked through, the rule alone and timed by the look-ahead; each reaches its exit without a hit.  The demonstration
builder (`teach`: the search, played again from the reset) on a way in and a way out: found, unhit, its rows those
of the way it played."""

import importlib.util
from pathlib import Path

import pytest

from .conftest import REPO_STATES_DIR, require_rom

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'experiments' / 'r40_d3_69_line.py'


@pytest.fixture(scope='module')
def line():
    require_rom()
    spec = importlib.util.spec_from_file_location('r40_d3_69_line', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_way_out_walks_the_line_without_a_strike(line):
    """Seed 0 from the cellar's exit spot ($60, $8D): N, E, N out by the door, no swing, no hit."""
    record = line.play(line.make_env(4, REPO_STATES_DIR), 'out', 0, 'd3first_3_69c_0')
    assert (record['ending'], record['hits'], record['swings']) == ('exit-correct', 0, 0), record


def test_the_way_in_swords_the_darknut_in_the_column_and_passes(line):
    """Seed 1 (8 bombs): the rule alone strikes a Darknut in Link's column (S) and walks on to the stairs unhit.
    Seed 0 (4 bombs) the same, timed by the look-ahead.  The seeds are the first of 0-39 that pass on the walked
    `d3first_3_69n_0` (round 44: one state per door, from the real walk; seeds 5 and 4 were chosen on round 40's
    rebuilt states, r40-states b9ebd4e9); a rebuilt entry state needs them chosen again (the script's `trace --pass
    in SEED --start STATE` shows the walk)."""
    alone = line.play(line.make_env(8, REPO_STATES_DIR), 'in', 1, 'd3first_3_69n_0')
    assert (alone['ending'], alone['hits'], alone['situation']) == ('exit-correct', 0, 'sworded'), alone
    looked = line.play(line.make_env(4, REPO_STATES_DIR), 'in', 0, 'd3first_3_69n_0', probe=line.make_env(4, REPO_STATES_DIR), depth=4)
    assert (looked['ending'], looked['hits'], looked['situation']) == ('exit-correct', 0, 'sworded'), looked
    assert looked['swings'] >= 1


@pytest.mark.parametrize('which, seed', [('out', 0), ('in', 3)])
def test_the_builder_finds_a_way_unhit_and_records_it(line, which, seed):
    """The search's way, played again from the visit's reset as one episode, leaves by the exit unhit; its rows are
    one a decision (a strike `REPEAT` times)."""
    from triforce.demos import DemoRecorder
    recorder = DemoRecorder()
    env = line.make_env(4 + 4 * (seed % 2), REPO_STATES_DIR)
    record = line.teach(env, which, seed, f'{line.STATES[which]}_0', recorder)   # round 44: one state per door
    assert (record['teacher'], record['ending'], record['hits']) == ('search', 'exit-correct', 0), record
    assert len(recorder.rows) == record['actions'] + (line.REPEAT['SWORD'] - 1) * record['swings']


def test_the_terminal_info_counts_the_episodes_hits(line):
    """`info['hits']` at the episode's end (what training logs by room): the steps whose hearts fell.  Seed 0: Link
    steps onto the floor and stands in the Darknuts' way until they kill him."""
    env = line.make_env(4, REPO_STATES_DIR)
    env.reset(seed=0, options={'start_state': 'd3first_3_69n_0'})
    health, hits = env.state.health, 0
    for text in ['MOVE S 4'] * 3 + ['MOVE E 4'] * 4 + ['WAIT 4'] * 200:
        _, _, terminated, truncated, info = env.step(text)
        hits += env.state.health < health
        health = env.state.health
        if terminated or truncated:
            break
    assert (info['ending'], info['hits']) == ('death', hits) and hits >= 2, info
