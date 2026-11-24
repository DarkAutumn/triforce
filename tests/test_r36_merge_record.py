"""``scripts/experiments/r36_merge_record.py`` with ``--state-route d3first`` records no start state outside the
route's namespace (round 38: the start-state pool drew ``d2first``'s bare-named harvest at ``harvested_fraction``
until it kept to its namespaces; the record's manifest, ``start_states_used``, found it)."""

import sys
from pathlib import Path

import pytest

from triforce import _native, paths, registry

from .conftest import REPO_STATES_DIR

ROOT = Path(__file__).resolve().parents[1]
ROUTE = 'd3first'


def script():
    sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))
    import r36_merge_record  # pylint: disable=import-outside-toplevel,import-error
    return r36_merge_record


def test_foreign_starts_names_every_state_outside_the_route():
    used = {'d3first_4_13w_0': 3, 'd3first_4_13w_h5032_16': 1, '4_13w_h5032_16': 2, '3_7cs_0': 1}
    assert script().foreign_starts(used, ROUTE) == ['3_7cs_0', '4_13w_h5032_16']
    assert script().foreign_starts(['d3first_0_77c_0'], ROUTE) == []


def test_a_route_part_with_both_routes_harvest_records_only_its_own(tmp_path):
    """boss-gleeok@d3first's harvested folder holds bare-named (``d2first``) and ``d3first_`` states; its task on
    the route's states draws harvested starts (``harvested_fraction``) only of ``d3first``.  The harvested states are
    the pool's (``paths.data_dir``: the checkout's folder while it has one, else the store's current snapshot)."""
    key = 'boss-gleeok@d3first'
    folder = paths.data_dir('harvested', REPO_STATES_DIR / 'harvested') / 'boss-gleeok'
    names = [p.stem for p in folder.glob('*.state')] if folder.is_dir() else []
    if not any(_native.state_namespace(n) is None for n in names) or not any(n.startswith('d3first_') for n in names):
        pytest.skip(f"needs boss-gleeok's harvested states of both routes ({folder})")
    try:
        registry.policy_checkpoint(key)
    except Exception:  # pylint: disable=broad-exception-caught
        pytest.skip(f"needs {key}'s teacher (the store's current team, which `triforce job test` gives the suites)")
    from triforce.lab.collect import Teachers, route_only, task_yaml  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist  # pylint: disable=import-outside-toplevel
    config = route_only(load_specialist(task_yaml(key)), ROUTE, REPO_STATES_DIR, tmp_path / 'tracks')
    config = config.replace(harvested_fraction=1.0)
    teachers = Teachers([key], 'cpu')
    meta = script().record(key, config, str(REPO_STATES_DIR), tmp_path / 'out', 2, 38, teachers, 2)
    used = meta['start_states_used']
    assert used and any('_h' in s for s in used), used
    assert script().foreign_starts(used, ROUTE) == []
