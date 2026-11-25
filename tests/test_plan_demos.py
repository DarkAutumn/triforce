"""scripts/experiments/r27_plan_demos.py: the start states it plays on each route (phase X's namespaces)."""

import importlib.util
from pathlib import Path

from .conftest import REPO_STATES_DIR

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location('r27_plan_demos',
                                                  ROOT / 'scripts' / 'experiments' / 'r27_plan_demos.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan_demos = _script()


def test_a_routes_stems_are_its_namespaced_states():
    names = plan_demos.state_names(['3_6bs', '4_40s'], 'd3first')
    assert names == ['d3first_3_6bs_0', 'd3first_4_40s_0']      # round 44: d3first has one state per door (chain 0)
    assert all((REPO_STATES_DIR / f'{name}.state').is_file() for name in names)


def test_d2first_and_no_route_keep_the_bare_names():
    names = plan_demos.state_names(['6_38s'], 'd2first')
    assert names == plan_demos.state_names(['6_38s']) == ['6_38s_0', '6_38s_1', '6_38s_2']
    assert all((REPO_STATES_DIR / f'{name}.state').is_file() for name in names)
