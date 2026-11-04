"""Every specialist's start states are clean pre-entry states with the route's progress (``triforce.entry_states``)."""

import json
import os

import pytest

from triforce.emulator import GAME_DIR
from triforce.entry_states import BASE_HEARTS, MANIFEST, audit_state, full_monster_counts, route_steps
from triforce.game_map import DEFAULT_ROUTE
from triforce.specialist import list_specialists, load_specialist

CONFIGURED = sorted({path for name in list_specialists() for path in load_specialist(name).resolve_start_states()})


@pytest.fixture(scope='module')
def manifest():
    with open(os.path.join(GAME_DIR, MANIFEST), encoding='utf-8') as file:
        return {entry['file']: entry for entry in json.load(file)}


def _route_hearts(entry: dict) -> int:
    """Max hearts after routed progress, including a completed off-route origin room."""
    route_step = entry['route_step']
    if route_step < 0 and 'from_step' in entry:
        route_step = entry['from_step'] + 1
    return BASE_HEARTS + sum(1 for step in route_steps(entry.get('route', DEFAULT_ROUTE))
                             if step.index < route_step and step.specialist.startswith('boss-'))


@pytest.mark.parametrize('path', CONFIGURED, ids=[os.path.basename(p) for p in CONFIGURED])
def test_configured_start_state_is_clean(session, manifest, path):
    row = audit_state(session, path, loads=2, full=full_monster_counts())
    assert row['named_room_matches'], f"loads into {row['room']}"
    assert row['pre_entry'] or row['full_count'] == 0, "saved after the room's monsters were created"
    assert row['full'], f"spawns {row['monsters_min']} of {row['full_count']} monsters"
    entry = manifest.get(os.path.basename(path))
    if entry is not None:
        assert row['hearts'] == _route_hearts(entry)
