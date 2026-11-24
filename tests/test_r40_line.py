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
    """Seed 3 (8 bombs): a Darknut crosses the column X $90 below Link; the rule alone strikes it and walks on to
    the stairs unhit.  Seed 4 (4 bombs) the same, timed by the look-ahead."""
    alone = line.play(line.make_env(8, REPO_STATES_DIR), 'in', 3, 'd3first_3_69n_0')
    assert (alone['ending'], alone['hits'], alone['situation']) == ('exit-correct', 0, 'sworded'), alone
    looked = line.play(line.make_env(4, REPO_STATES_DIR), 'in', 4, 'd3first_3_69n_1', probe=line.make_env(4, REPO_STATES_DIR), depth=4)
    assert (looked['ending'], looked['hits'], looked['situation']) == ('exit-correct', 0, 'sworded'), looked
    assert looked['swings'] >= 1


@pytest.mark.parametrize('which, seed', [('out', 0), ('in', 3)])
def test_the_builder_finds_a_way_unhit_and_records_it(line, which, seed):
    """The search's way, played again from the visit's reset as one episode, leaves by the exit unhit; its rows are
    one a decision (a strike `REPEAT` times)."""
    from triforce.demos import DemoRecorder
    recorder = DemoRecorder()
    env = line.make_env(4 + 4 * (seed % 2), REPO_STATES_DIR)
    record = line.teach(env, which, seed, f'{line.STATES[which]}_{seed % 3}', recorder)
    assert (record['teacher'], record['ending'], record['hits']) == ('search', 'exit-correct', 0), record
    assert len(recorder.rows) == record['actions'] + (line.REPEAT['SWORD'] - 1) * record['swings']
