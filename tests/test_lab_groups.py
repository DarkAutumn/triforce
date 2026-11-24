"""The generation's teachers as data (``triforce/groups.yaml``, ``triforce.lab.groups``) and ``lab.train --group``."""

import json

import pytest
import torch

from triforce import _native, registry
from triforce.lab import groups, recipe, train
from triforce.lab.collect import parse_families, task_yaml
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR
from .networks import native_spec

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


GENERATION = {'arch': 'hitnet', 'arch_config': {'tokens': ['type', 'features', 'rel', 'threat', 'chase'],
                                                'feed': True, 'room': True, 'hidden_dim': 768},
              'aux': ['hit=hit:0.5'], 'try_others': '4:0.25'}


def test_the_files_network_block_is_every_teachers_and_group_applies_it(capsys):
    """The owner's width (768 everywhere) and the lead's pins live in the file: ``--group NAME`` alone trains them."""
    assert all(group.network == GENERATION for group in groups.load().values())
    args = train.parse_args(['--name', 'rooms', '--group', 'rooms'])
    sources = train.apply_network(args, train.expand_group(args))
    assert (args.arch, json.loads(args.arch_config), args.aux, args.try_others) == (
        'hitnet', GENERATION['arch_config'], ['hit=hit:0.5'], '4:0.25')
    assert set(sources.values()) == {'group file'} and '[lab] --group' not in capsys.readouterr().out
    # the command line wins, by key for --arch-config, and one line names what it changed
    args = train.parse_args(['--name', 'rooms', '--group', 'rooms', '--arch-config', '{"hidden_dim": 256}', '--aux',
                             'hit=hit:1.0', '--try-others', '4:0.25'])
    sources = train.apply_network(args, train.expand_group(args))
    assert json.loads(args.arch_config) == {**GENERATION['arch_config'], 'hidden_dim': 256}
    assert args.aux == ['hit=hit:1.0'] and args.try_others == '4:0.25'
    assert sources == {'arch': 'group file', 'arch_config': 'command line', 'aux': 'command line',
                       'try_others': 'command line'}
    out = [line for line in capsys.readouterr().out.splitlines() if line.startswith('[lab] --group')]
    assert out == ["[lab] --group rooms: the command line overrides the group file's arch_config.hidden_dim 768 -> "
                   '256; aux ["hit=hit:0.5"] -> ["hit=hit:1.0"]']
    # another architecture takes none of the file's settings
    args = train.parse_args(['--name', 'rooms', '--group', 'rooms', '--arch', 'wide'])
    train.apply_network(args, train.expand_group(args))
    assert (args.arch, args.arch_config) == ('wide', '{}')
    assert "arch_config: the file's is hitnet's, not used" in capsys.readouterr().out


def test_a_teacher_overrides_a_key_and_a_file_without_the_block_behaves_as_before(tmp_path):
    path = tmp_path / 'groups.yaml'
    path.write_text("route: d3first\nnetwork: {arch: hitnet, arch_config: {feed: true, hidden_dim: 768}, "
                    "aux: [hit=hit:0.5]}\nteachers:\n  a: {members: [zol@d3first]}\n"
                    "  b: {members: [vire@d3first], network: {arch_config: {hidden_dim: 512}, try_others: '2:0.5'}}\n")
    loaded = groups.load(path)
    assert loaded['a'].network == {'arch': 'hitnet', 'arch_config': {'feed': True, 'hidden_dim': 768},
                                   'aux': ['hit=hit:0.5']}
    assert loaded['b'].network == {'arch': 'hitnet', 'arch_config': {'feed': True, 'hidden_dim': 512},
                                   'aux': ['hit=hit:0.5'], 'try_others': '2:0.5'}
    path.write_text("route: d3first\nteachers:\n  a: {members: [zol@d3first]}\n")
    bare = groups.get('a', path)
    assert bare.network == {} and 'network' not in bare.record()
    args = train.parse_args(['--name', 'x', '--group', 'a'])
    sources = train.apply_network(args, bare)
    plain = train.parse_args(['--name', 'x', '--families', 'zol'])
    assert train.apply_network(plain, None) is None
    assert (args.arch, args.arch_config, args.aux, args.try_others) == (
        plain.arch, plain.arch_config, plain.aux, plain.try_others) == ('zelda', '{}', [], None)
    assert set(sources.values()) == {'default'}
    path.write_text("route: d3first\nnetwork: {width: 768}\nteachers:\n  a: {members: [zol@d3first]}\n")
    with pytest.raises(ValueError, match="unknown keys \\['width'\\]"):
        groups.load(path)


def test_the_generations_hitnet_builds_at_768_and_runs_a_forward_pass():
    network = recipe.create_network(GENERATION['arch'], GENERATION['arch_config'])
    assert network.parameter_count() == 4_960_388
    obs = {key: (torch.randint(0, spec['vocab'], (2, *spec['shape'])) if spec['vocab']
                 else torch.rand(2, *spec['shape'])) for key, spec in native_spec().items() if key != 'action_mask'}
    heads, _ = network.heads(obs)
    assert heads.kind.shape == (2, 11) and heads.value.shape == (2,) and heads.aux['hit'].shape == (2, 352)
