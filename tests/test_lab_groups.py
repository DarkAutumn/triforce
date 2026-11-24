"""The generation's teachers as data (``triforce/groups.yaml``, ``triforce.lab.groups``) and ``lab.train --group``."""

import pytest

from triforce import _native, registry
from triforce.lab import groups, train
from triforce.lab.collect import parse_families, task_yaml
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR

# The lead's pins for 2.0.1.0 (four groups, seven bosses each its own teacher).
TEACHERS = {
    'overworld': ['ow-start', 'ow-tektite', 'ow-leever', 'ow-moblin', 'ow-octorok', 'ow-zora', 'ow-peahat', 'ow-lynel',
                  'ow-ghini', 'ow-boulder'],
    'walkers': ['darknut', 'gibdo', 'stalfos-keese-gel', 'zol', 'like-like', 'vire', 'goriya', 'd2-rope',
                'dodongo-crossing', 'moldorm'],
    'others': ['wizzrobe', 'pols-voice', 'wallmaster', 'lanmola', 'patra'],
    'rooms': ['equipment-room', 'trap-room'],
    **{boss: [boss] for boss in ('boss-manhandla', 'boss-gleeok', 'boss-aquamentus', 'boss-dodongo', 'boss-digdogger',
                                 'boss-gohma', 'ganon')},
}


def test_the_file_names_every_teacher_and_every_family_of_the_route_once():
    loaded = groups.load()
    assert {name: group.families for name, group in loaded.items()} == TEACHERS
    assert all(group.route == 'd3first' and all(key.endswith('@d3first') for key in group.members)
               for group in loaded.values())
    families = [family for group in loaded.values() for family in group.families]
    assert len(families) == len(set(families))                          # no family in two teachers
    route = {step['specialist'] for step in _native.route_steps('d3first') if step['specialist']}
    assert set(families) == route                                       # every family of the route in one


def test_every_members_task_file_exists_and_loads():
    for group in groups.load().values():
        for key in group.members:
            task = task_yaml(key)
            config = load_specialist(task)
            assert config.name == registry.split_policy(key)[0], (key, task)


def test_every_generation_task_loads_on_the_route_as_it_does_without_it(tmp_path):
    """``--group`` implies ``--state-route d3first`` (``lab.collect.route_only``): every member's task loads there
    (its inputs checked) and keeps its curriculum settings, the ``untracked`` declaration among them.  The lead hit
    the fault this pins (Nov 24): ``route_only`` rebuilt the curriculum block from ``enabled`` and the link folder
    alone, so a task declaring states untracked (zol@d3first's Dungeon 5/6/9 states; its 3:0x4b too until Tracks41
    gave it tracks) was refused."""
    for group in groups.load().values():
        for key in group.members:
            bare, = parse_families([key], 1, states_dir=REPO_STATES_DIR)
            routed, = parse_families([key], 1, state_route=group.route, states_dir=REPO_STATES_DIR,
                                     tracks_dir=tmp_path)
            assert list(routed.config.state_routes) == [group.route], key
            mine, theirs = routed.config.to_dict()['curriculum'], bare.config.to_dict()['curriculum']
            if mine['enabled']:
                assert {k: v for k, v in mine.items() if k != 'trajectories'} == {
                    k: v for k, v in theirs.items() if k != 'trajectories'}, key
    zol, = parse_families(['zol@d3first'], 1, state_route='d3first', states_dir=REPO_STATES_DIR, tracks_dir=tmp_path)
    assert 'd3first_5_57w_[0-9]' in zol.config.to_dict()['curriculum']['untracked']


@pytest.mark.parametrize('text, message', [
    ("route: d3first\nteachers:\n  a: {members: [zol@d3first]}\n  b: {members: [zol@d3first, vire@d3first]}\n",
     'zol is in two teachers'),
    ("route: d3first\nteachers:\n  a: {members: [zol@d2first]}\n", 'not on route d3first'),
    ("route: d3first\nteachers:\n  a: {members: [zol]}\n", 'write it zol@d3first'),
    ("route: d3first\nteachers:\n  a+b: {members: [zol@d3first]}\n", "without '\\+'"),
    ("route: d3first\nteachers:\n  a: {members: [zol@d3first], shares: {vire@d3first: 2}}\n", 'name no member'),
    ("route: d3first\nteachers:\n  a: {members: []}\n", 'no members'),
])
def test_a_malformed_file_is_refused_naming_what_to_change(tmp_path, text, message):
    path = tmp_path / 'groups.yaml'
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        groups.load(path)


def test_shares_are_read_by_member(tmp_path):
    path = tmp_path / 'groups.yaml'
    path.write_text("route: d3first\nteachers:\n  pair: {members: [zol@d3first, vire@d3first], "
                    "shares: {vire@d3first: 3}}\n")
    group = groups.get('pair', path)
    assert group.shares == {'vire@d3first': 3.0} and group.record()['members'] == ['zol@d3first', 'vire@d3first']
    with pytest.raises(SystemExit, match='no such teacher'):
        groups.get('nope', path)


def test_group_expands_to_its_members_on_its_route():
    args = train.parse_args(['--name', 'rooms', '--group', 'rooms'])
    group = train.expand_group(args)
    assert group.name == 'rooms'
    assert args.families == ['equipment-room@d3first', 'trap-room@d3first'] and args.state_route == 'd3first'
    with pytest.raises(SystemExit, match='one or the other'):
        train.expand_group(train.parse_args(['--name', 'x', '--group', 'rooms', '--families', 'zol']))
    with pytest.raises(SystemExit, match='not --state-route d2first'):
        train.expand_group(train.parse_args(['--name', 'x', '--group', 'rooms', '--state-route', 'd2first']))
    with pytest.raises(SystemExit, match='--families KEY'):
        train.expand_group(train.parse_args(['--name', 'x']))
    args = train.parse_args(['--name', 'x', '--families', 'zol'])
    assert train.expand_group(args) is None and args.families == ['zol']
