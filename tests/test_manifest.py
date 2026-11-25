"""Round 40: a run's manifest (``triforce.manifest``): what it records, by content and without a checkout's own
path, and ``diff`` / ``triforce manifest diff`` between two runs, also of records made before manifests."""

import json
from pathlib import Path

import pytest

from triforce import manifest
from triforce.paths import SPECIALISTS_DIR
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR, require_rom

ROOT = Path(__file__).resolve().parents[1]


def test_labels_are_relative_to_the_checkout_a_path_lies_in(tmp_path):
    assert manifest.label(ROOT / 'specialists' / 'zol.yaml') == 'specialists/zol.yaml'
    assert manifest.label(Path.home() / 'elsewhere' / 'x') == '~/elsewhere/x'
    assert manifest.label(None) is None


def test_a_tree_digest_is_by_name_and_content(tmp_path):
    for name, text in (('a.state', 'one'), ('b.state', 'two')):
        (tmp_path / name).write_text(text)
    first = manifest.tree(sorted(tmp_path.glob('*.state')))
    assert first['count'] == 2 and first == manifest.tree(reversed(sorted(tmp_path.glob('*.state'))))
    (tmp_path / 'b.state').write_text('changed')
    assert manifest.tree(tmp_path.glob('*.state'))['sha256'] != first['sha256']


def test_the_data_section_counts_and_hashes_each_input(tmp_path):
    require_rom()
    config = load_specialist(str(ROOT / 'specialists' / 'd3first' / 'zol.yaml'))
    harvested = tmp_path / 'harvested'
    harvested.mkdir()
    (harvested / 'd3first_3_7cs_h1_0.state').write_text('x')
    config = config.replace(harvested_states=str(harvested), harvested_fraction=0.25)
    data = manifest.data(config, REPO_STATES_DIR)
    assert set(data['start_states']['patterns']) == set(config.start_states)
    assert data['start_states']['count'] == sum(p['count'] for p in data['start_states']['patterns'].values())
    assert data['harvested']['count'] == 1 and data['harvested']['fraction'] == 0.25
    assert data['states_dir'] == 'triforce/states'


def test_diff_lists_every_difference_by_path_and_group():
    a = {'code': {'commit': 'x'}, 'data': {'harvested': {'count': 0}}, 'arguments': {'seed': 1}, 'networks': {}}
    b = {'code': {'commit': 'y'}, 'data': {'harvested': {'count': 3396}}, 'arguments': {'seed': 1}}
    assert manifest.diff(a, b) == [{'path': 'code.commit', 'a': 'x', 'b': 'y'},
                                   {'path': 'data.harvested.count', 'a': 0, 'b': 3396}]
    assert manifest.group('rom.sha1') == 'data' and manifest.group('job.id') == 'run'
    assert manifest.diff(a, a) == []


def test_records_from_before_manifests_are_read_and_say_what_they_cannot(tmp_path, capsys):
    """Two run.json files as round 40's arms left them (one trained with no harvested states, one with 3396)."""
    def run(name: str, states: int, load, sil: float) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        args = {'specialist': 'specialists/d3first/zol.yaml', 'load': load, 'seed': 401, 'sil_coeff': sil,
                'run_id': name}
        (directory / 'run.json').write_text(json.dumps({
            'args': args, 'specialist': 'zol', 'run_id': name, 'source': load, 'start_iteration': 0,
            'curriculum': None, 'curriculum_tracks': None,
            'harvested': {'dir': 'triforce/states/harvested/zol', 'states': states, 'fraction': 0.25}}))
        return directory
    a, b = run('cont', 0, None, 0.1), run('fresh', 3396, None, 0.0)
    assert manifest.main(['diff', str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert 'data.harvested.count: 0 -> 3396' in out and 'arguments.sil_coeff: 0.1 -> 0.0' in out
    assert 'unknown in A (a record from before manifests): code, rom' in out
    assert manifest.main(['diff', str(tmp_path / 'nothing'), str(b)]) == 2


def test_the_same_recipe_differs_only_in_the_run_and_the_excepted_keys():
    base = {'schema': 1, 'kind': 'train', 'code': {'commit': 'c', 'module': {'built': 't1'}},
            'data': {'harvested': {'count': 3396, 'sha256': 'h'}},
            'arguments': {'seed': 401, 'run_id': 'a', 'sil_coeff': 0.1, 'iterations': 244},
            'seeds': {'pool': 1}, 'job': {'id': 1}}
    arm = json.loads(json.dumps(base))
    arm['arguments'].update(seed=402, run_id='b')
    arm['seeds'], arm['job'], arm['code']['module']['built'] = {'pool': 2}, {'id': 2}, 't2'
    assert manifest.same_recipe(arm, base) == ([], [])
    arm['arguments']['sil_coeff'] = 0.0
    assert [d['path'] for d in manifest.same_recipe(arm, base)[0]] == ['arguments.sil_coeff']
    assert manifest.same_recipe(arm, base, ['sil_coeff']) == ([], [])
    arm['data']['harvested'] = {'count': 0, 'sha256': 'e'}
    assert [d['path'] for d in manifest.same_recipe(arm, base, ['sil_coeff'])[0]] == [
        'data.harvested.count', 'data.harvested.sha256']
    assert manifest.same_recipe(arm, base, ['sil_coeff', 'harvested']) == ([], [])


def test_against_a_record_from_before_manifests_what_it_never_said_is_unknown():
    legacy = manifest.legacy_train({'args': {'specialist': 'zol', 'seed': 1, 'sil_coeff': 0.1}, 'specialist': 'zol',
                                    'harvested': {'dir': None, 'states': 0, 'fraction': 0.25}})
    mine = {'schema': 1, 'code': {'commit': 'c'}, 'task': {'family': 'zol', 'yaml': 'specialists/zol.yaml'},
            'data': {'harvested': {'dir': None, 'count': 3396, 'fraction': 0.25, 'sha256': 'h'}},
            'arguments': {'sil_coeff': 0.1, 'curriculum_unrecorded': None, 'specialist': 'zol'}}
    differences, unknown = manifest.same_recipe(mine, legacy)
    assert [d['path'] for d in differences] == ['data.harvested.count']
    assert {d['path'] for d in unknown} >= {'code.commit', 'data.harvested.sha256'}


def _run_dir(tmp_path, name, recipe):
    folder = tmp_path / name
    folder.mkdir()
    (folder / 'run.json').write_text(json.dumps({'manifest': recipe}))
    return folder


def _recipe(**arguments):
    return {'schema': 1, 'kind': 'train', 'arguments': {'sil_coeff': 0.1, 'iterations': 10, **arguments},
            'data': {'harvested': {'count': 3396}}, 'run': {'id': 'x'}}


def test_same_recipe_as_refuses_a_training_that_differs_beyond_its_exceptions(tmp_path):
    from triforce import train
    reference = _run_dir(tmp_path, 'ref', _recipe())
    train.check_same_recipe(_recipe(iterations=20), str(reference), ['iterations'])
    with pytest.raises(train.RecipeMismatch) as error:
        train.check_same_recipe(_recipe(iterations=20, sil_coeff=0.0), str(reference), ['iterations'])
    assert 'arguments.sil_coeff: 0.0 (this run) vs 0.1' in str(error.value)


def test_same_team_as_refuses_a_chain_whose_other_networks_differ(tmp_path):
    from triforce import cli
    from triforce.runner import RunnerError
    team = {'networks': {'zol': {'sha1': 'a'}, 'gohma': {'sha1': 'b'}}, 'task': {'configs': {}}}
    output = tmp_path / 'base.jsonl'
    (tmp_path / 'base.jsonl.meta.json').write_text(json.dumps({'manifest': team}))
    mine = {'networks': {'zol': {'sha1': 'c'}, 'gohma': {'sha1': 'b'}}, 'task': {'configs': {}}}
    cli.check_same_team(mine, str(output), ['zol'])
    with pytest.raises(RunnerError) as error:
        cli.check_same_team(mine, str(output), [])
    assert 'networks.zol.sha1' in str(error.value)


def test_a_task_file_extends_its_family_with_the_lines_it_changes(tmp_path):
    from triforce.specialist import load_specialist
    base = SPECIALISTS_DIR / 'ow-start.yaml'
    child = tmp_path / 'exp.yaml'
    child.write_text(f"extends: {base}\nmax_steps: 123\n")
    config = load_specialist(str(child))
    assert config.max_steps == 123 and config.name == load_specialist('ow-start').name
    recorded = manifest.task(config, child)
    assert recorded['extends'][0]['sha256'] == manifest.file_sha256(base)
    assert recorded['overrides'] == {'max_steps': 123}
    loop = tmp_path / 'loop.yaml'
    loop.write_text("extends: loop.yaml\n")
    with pytest.raises(ValueError, match='extends: a loop'):
        load_specialist(str(loop))


def _two_bases(tmp_path, common: str) -> Path:
    """A task extending ow-start's yaml and a common block (`extends: [base, common]`, as the generation's d3first
    tasks extend their family's yaml and `_generation.yaml`), with a key of its own."""
    (tmp_path / 'common.yaml').write_text(common)
    child = tmp_path / 'exp.yaml'
    child.write_text(f"extends: [{SPECIALISTS_DIR / 'ow-start.yaml'}, common.yaml]\nmax_steps: 77\n")
    return child


def test_a_task_file_extends_a_list_of_bases_merged_in_order(tmp_path):
    """Round 40 (the generation's tasks): the bases merge in order, each over the one before, the file's own keys
    last; mappings key by key."""
    child = _two_bases(tmp_path, "obs: {threat: true}\nreward: {aimless_penalty: 0.002}\nmax_steps: 5\n"
                                 "warm_start: null\n")
    config = load_specialist(str(child))
    base = load_specialist('ow-start')
    assert config.obs.threat and not config.obs.chase and config.obs.target == base.obs.target
    assert config.reward.aimless_penalty == 0.002 and config.reward.pbrs == base.reward.pbrs
    assert config.max_steps == 77                                  # the file's own key over both bases
    assert config.rooms == base.rooms
    recorded = manifest.task(config, child)
    assert [e['yaml'] for e in recorded['extends']] == [manifest.label(SPECIALISTS_DIR / 'ow-start.yaml'),
                                                        manifest.label(tmp_path / 'common.yaml')]
    assert [e['sha256'] for e in recorded['extends']] == [manifest.file_sha256(SPECIALISTS_DIR / 'ow-start.yaml'),
                                                          manifest.file_sha256(tmp_path / 'common.yaml')]
    assert recorded['overrides'] == {'max_steps': 77}


def test_a_list_of_bases_refuses_a_loop_a_missing_base_and_an_empty_list(tmp_path):
    loop = tmp_path / 'loop.yaml'
    loop.write_text(f"extends: [{SPECIALISTS_DIR / 'ow-start.yaml'}, loop.yaml]\n")
    with pytest.raises(ValueError, match=r'loop\.yaml: extends: a loop'):
        load_specialist(str(loop))
    missing = tmp_path / 'missing.yaml'
    missing.write_text(f"extends: [{SPECIALISTS_DIR / 'ow-start.yaml'}, nowhere.yaml]\n")
    with pytest.raises(ValueError, match=r'missing\.yaml: extends: nowhere\.yaml: no such task file'):
        load_specialist(str(missing))
    empty = tmp_path / 'empty.yaml'
    empty.write_text("extends: []\n")
    with pytest.raises(ValueError, match='a non-empty list'):
        load_specialist(str(empty))
    # A base two entries share is no loop.
    (tmp_path / 'a.yaml').write_text(f"extends: {SPECIALISTS_DIR / 'ow-start.yaml'}\nmax_steps: 11\n")
    (tmp_path / 'b.yaml').write_text(f"extends: {SPECIALISTS_DIR / 'ow-start.yaml'}\none_room: false\n")
    both = tmp_path / 'both.yaml'
    both.write_text("extends: [a.yaml, b.yaml]\n")
    config = load_specialist(str(both))
    # b's whole merged task lies over a's: ow-start's own max_steps from b covers a's 11.
    assert config.max_steps == load_specialist('ow-start').max_steps and not config.one_room


def test_manifest_diff_shows_a_change_in_the_common_block(tmp_path, capsys):
    sides = []
    for name, penalty in (('a', 0.002), ('b', 0.005)):
        folder = tmp_path / name
        folder.mkdir()
        child = _two_bases(folder, f"reward: {{aimless_penalty: {penalty}}}\n")
        task = manifest.task(load_specialist(str(child)), child)
        (folder / 'run.json').write_text(json.dumps({'manifest': {'task': task}}))
        sides.append(folder)
    paths = [d['path'] for d in manifest.diff(manifest.read(sides[0]), manifest.read(sides[1]))]
    assert 'task.effective.reward.aimless_penalty' in paths and 'task.extends' in paths
    assert manifest.main(['diff', str(sides[0]), str(sides[1])]) == 0
    out = capsys.readouterr().out
    assert 'task.effective.reward.aimless_penalty: 0.002 -> 0.005' in out
