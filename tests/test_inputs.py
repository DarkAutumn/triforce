"""Round 40 (the owner: missing is an error, never "none"): every input a run declares that resolves to nothing stops
it before it starts, naming where it looked; where none is a legitimate recipe the task declares it."""

import fnmatch
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from triforce import build, job, paths, registry, train
from triforce.inputs import (MissingInput, check_curriculum, check_harvested, check_start_states, check_task,
                             declaration, track_starts, untracked)
from triforce.runner import Policies, Route, RunnerError, check_configs, check_team
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR, require_rom
from .networks import scripted

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def zol():
    """zol@d3first's task (ten start-state globs, harvested states at 0.25, a curriculum)."""
    require_rom()
    return load_specialist(str(ROOT / 'specialists' / 'd3first' / 'zol.yaml'))


def _harvested(directory: Path, names) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f'{name}.state').write_bytes(b'x')
    return directory


def test_a_start_state_glob_that_matches_nothing_is_refused(zol):
    matched = check_start_states(zol, REPO_STATES_DIR)
    assert all(matched.values()) and len(matched) == len(zol.start_states)
    with pytest.raises(MissingInput, match=r"\['9_99\*_\[0-9\]'\] match no state file under"):
        check_start_states(zol.replace(start_states=[*zol.start_states, '9_99*_[0-9]']), REPO_STATES_DIR)


def test_harvested_states_a_task_draws_must_be_there(zol, tmp_path):
    empty = zol.replace(harvested_states=str(tmp_path / 'none'))
    with pytest.raises(MissingInput, match=f"draws 25% of its resets from harvested start states and {tmp_path}/none"):
        check_harvested(empty, REPO_STATES_DIR)
    # Another route's (bare names) don't count: zol@d3first draws its own namespace's only.
    other = _harvested(tmp_path / 'bare', ['3_7cs_h1_0'])
    with pytest.raises(MissingInput):
        check_harvested(zol.replace(harvested_states=str(other)), REPO_STATES_DIR)
    own = _harvested(tmp_path / 'own', ['d3first_3_7cs_h1_0'])
    assert len(check_harvested(zol.replace(harvested_states=str(own)), REPO_STATES_DIR)) == 1
    assert check_harvested(zol.replace(harvested_fraction=0.0, harvested_states=str(tmp_path / 'none'))) == []


def _tracks(directory: Path, starts) -> Path:
    for index, start in enumerate(starts):
        track = directory / f'{start}~t{index}'
        track.mkdir(parents=True)
        (track / 'trajectory.json').write_text(json.dumps({'start_state': f'{start}.state'}))
    return directory


def test_every_seed_state_has_a_track_or_is_declared_untracked(zol, tmp_path):
    seeds = sorted(Path(p).stem for p in zol.resolve_start_states(str(REPO_STATES_DIR)))
    with pytest.raises(MissingInput, match='holds no track'):
        check_curriculum(zol, tmp_path / 'nothing', REPO_STATES_DIR)
    tracked = _tracks(tmp_path / 'tracks', seeds[:-3])
    assert set(track_starts(tracked).values()) == set(seeds[:-3])
    bare = zol.replace(curriculum={'enabled': True})
    with pytest.raises(MissingInput, match=r"3 of \d+ seed states have no curriculum track") as refused:
        check_curriculum(bare, tracked, REPO_STATES_DIR)
    glob = seeds[-1][:-1] + '[0-9]'
    assert f"curriculum.untracked: ['{glob}']" in str(refused.value)
    declared = zol.replace(curriculum={'enabled': True, 'untracked': [glob]})
    assert len(check_curriculum(declared, tracked, REPO_STATES_DIR)) == len(seeds) - 3


def test_a_declaration_never_covers_a_tracked_state():
    seeds = ['a_0', 'a_1', 'a_2', 'b_0', 'b_1']
    assert declaration(['a_1', 'b_0', 'b_1'], seeds) == ['a_1', 'b_[0-9]']


def test_the_shipped_tasks_declare_every_input_they_draw():
    """Every task of specialists/ passes its checks on this checkout's data (the declarations of round 40)."""
    require_rom()
    for path in sorted((ROOT / 'specialists').rglob('*.yaml')):
        config = load_specialist(str(path))
        trajectories = (str(config.curriculum.trajectories or paths.data_dir('trajectories') / config.name)
                        if config.curriculum.enabled else None)
        if trajectories is not None and not Path(trajectories).is_dir():
            pytest.skip(f"{trajectories}: this checkout has no trajectories")
        check_task(config, REPO_STATES_DIR, trajectories)


# Round 40 (Tracks40): the seed states whose stale tracks left the generation's curricula and that the explorer did
# not solve in 3000 iterations (with darknut's 0x59 from N at 4 bombs, States40's), per family, beside the same
# rooms' chains that do have a track. Both of the family's tasks draw them (its d3first part's and its own).
LEFT_WITHOUT_A_TRACK = {
    'darknut': ({'d3first_3_59n_1', 'd3first_3_59n_2', 'd3first_3_5bw_0', 'd3first_3_5bw_2'},
                {'d3first_3_59n_0', 'd3first_3_5bw_1'}),
    'zol': ({'d3first_3_4bs_0', 'd3first_3_4bs_1'}, {'d3first_3_4bs_2'}),
    'vire': ({f'd3first_4_{room}_{c}' for room in ('12n', '31w') for c in range(3)}, set()),
    'like-like': ({f'd3first_4_32w_{c}' for c in range(3)}, {'d3first_4_32c_0'}),
    'boss-gleeok': ({f'd3first_4_13w_{c}' for c in range(3)}, set()),
    'goriya': ({'d3first_1_23s_1', 'd3first_1_23s_2', 'd3first_2_1es_1', *(f'd3first_7_1bw_{c}' for c in range(3))},
               {'d3first_1_23s_0', 'd3first_2_1es_0', 'd3first_2_1es_2'}),
    'boss-digdogger': ({f'd3first_7_1cw_{c}' for c in range(3)}, {'d3first_7_39s_0', 'd3first_5_24e_0'}),
}


def test_the_d3first_tasks_declare_the_rooms_left_without_a_track():
    """Round 40 (the lead, Nov 24): 0:0x76's explored tracks walk across and kill, so ow-tektite@d3first trains its
    three seeds without one; darknut@d3first's 0x59 from N with 4 bombs (chains 1 and 2) had no explored track; and
    Tracks40's states without an explored track (`LEFT_WITHOUT_A_TRACK`). The tasks load, and the declarations cover
    exactly those seeds: never a chain of the same room that has a track."""
    require_rom()
    tektite = load_specialist(str(ROOT / 'specialists' / 'd3first' / 'ow-tektite.yaml'))
    seeds = sorted(Path(p).stem for p in tektite.resolve_start_states(str(REPO_STATES_DIR)))
    assert [s for s in seeds if s.startswith('d3first_0_76e_')] == [f'd3first_0_76e_{c}' for c in range(3)]
    assert declaration([f'd3first_0_76e_{c}' for c in range(3)], seeds) == untracked(tektite)
    for family, (left, tracked) in LEFT_WITHOUT_A_TRACK.items():
        for path in (ROOT / 'specialists' / 'd3first' / f'{family}.yaml', ROOT / 'specialists' / f'{family}.yaml'):
            config = load_specialist(str(path))
            if not config.curriculum.enabled:           # boss-digdogger@d3first trains without a curriculum
                continue
            seeds = {Path(p).stem for p in config.resolve_start_states(str(REPO_STATES_DIR))}
            declared = untracked(config)
            assert left | tracked <= seeds, path
            assert all(any(fnmatch.fnmatchcase(s, g) for g in declared) for s in left), path
            assert not any(fnmatch.fnmatchcase(s, g) for s in tracked for g in declared), path


def test_a_warm_start_names_a_checkpoint_or_a_family_with_one(tmp_path):
    (tmp_path / 'empty-run').mkdir()
    with pytest.raises(MissingInput, match='a directory without ckpt.pt'):
        train.resolve_warm_start(str(tmp_path / 'empty-run'), tmp_path)
    with pytest.raises(MissingInput, match='no checkpoint file or run dir there, and no family of that name'):
        train.resolve_warm_start(str(tmp_path / 'models' / 'zol' / 'r34d3-zol-s34'), tmp_path)
    with pytest.raises(MissingInput, match='no checkpoint file or run dir there'):
        train.resolve_warm_start('zoll', tmp_path)
    with pytest.raises(MissingInput, match="No promoted model for specialist 'zol'"):
        train.resolve_warm_start('zol', tmp_path)
    saved = registry.save_checkpoint('zol', 'r1', scripted('WAIT 8'), root=tmp_path)
    assert train.resolve_warm_start(str(saved.parent), tmp_path) == saved.parent


def test_train_refuses_a_missing_input_before_it_waits(tmp_path):
    """In the wrapper's dry call (``$TRIFORCE_JOB_ESTIMATE``), so the job never joins the line."""
    result = subprocess.run([sys.executable, '-m', 'triforce.train', '--specialist', 'zol', '--device', 'cpu',
                             '--load', str(tmp_path / 'nowhere'), '--models-dir', str(tmp_path)],
                            capture_output=True, text=True, check=False,
                            env={**os.environ, 'TRIFORCE_JOB_ESTIMATE': str(tmp_path / 'work.json')})
    assert result.returncode == 2 and not (tmp_path / 'work.json').exists()
    assert f"warm start {tmp_path}/nowhere" in result.stderr, result.stderr


def test_a_continuation_of_a_run_without_its_curriculum_record_must_say_what_to_do(zol, tmp_path):
    loaded = tmp_path / 'old'
    loaded.mkdir()
    (loaded / 'curriculum.json').write_text('{}')
    (loaded / 'run.json').write_text(json.dumps({'args': {}}))
    tracks = _tracks(tmp_path / 'tracks', ['d3first_3_7cs_0'])
    out = tmp_path / 'new'
    out.mkdir()
    args = Namespace(no_curriculum=False, curriculum=str(tracks), curriculum_fraction=None, curriculum_threshold=None,
                     curriculum_unrecorded=None)
    with pytest.raises(MissingInput, match='--curriculum-unrecorded resume'):
        train.curriculum_spec(args, zol, out, loaded)
    args.curriculum_unrecorded = 'resume'
    train.curriculum_spec(args, zol, out, loaded)
    assert (out / 'curriculum.json').read_text() == '{}'
    args.curriculum_unrecorded = 'fresh'
    train.curriculum_spec(args, zol, out, loaded)
    assert not (out / 'curriculum.json').exists()


def test_the_lab_refuses_missing_harvested_states(tmp_path):
    from triforce.lab.collect import parse_families  # pylint: disable=import-outside-toplevel
    require_rom()
    text = (ROOT / 'specialists' / 'd3first' / 'zol.yaml').read_text()
    task = tmp_path / 'zol.yaml'
    task.write_text(text + f"harvested_states: {tmp_path / 'none'}\n")
    with pytest.raises(SystemExit, match='holds none'):
        parse_families([f'zol@d3first={task}'], 1, curriculum=False, states_dir=REPO_STATES_DIR)


def test_a_chain_refuses_a_policy_its_runs_reach_without_a_network():
    require_rom()
    route = Route.load('d3first', start_from='3_7cs_0', stop_step=24, states_dir=REPO_STATES_DIR)
    played = {step['specialist'] for step in route.steps_in_play()}
    assert 'zol' in played and 'ganon' not in played
    check_team(route, SimpleNamespace(names=['zol@d3first'], missing={'ganon': 'no promoted run'}))
    with pytest.raises(RunnerError, match=r"zol \(step \d+: none\)"):
        check_team(route, SimpleNamespace(names=[], missing={'zol': 'none'}))


def test_a_specialist_config_the_runs_never_play_is_refused():
    require_rom()
    route = Route.load('d3first', start_from='3_7cs_0', stop_step=24, states_dir=REPO_STATES_DIR)
    check_configs(route, {'zol': load_specialist('zol')})
    with pytest.raises(RunnerError, match=r"--specialist-config for \['ganon'\]"):
        check_configs(route, {'ganon': load_specialist('ganon')})


def test_an_override_that_drops_a_hit_head_must_be_declared(tmp_path):
    team = tmp_path / 'models'
    with_head = registry.checkpoint_path('zol', 'head', team)
    with_head.parent.mkdir(parents=True)
    torch.save({'specialist': 'zol', 'state_dict': {'hit.hit_head.weight': torch.zeros(1)}}, with_head)
    (team / 'zol' / 'promoted').write_text('head\n')
    bare = registry.save_checkpoint('zol', 'bare', scripted('WAIT 8'), root=tmp_path / 'other')
    with pytest.raises(RunnerError, match='has no hit head and replaces zol .*declare it with --headless zol'):
        Policies(team, {'zol': str(bare)}, names=['zol'])
    assert Policies(team, {'zol': str(bare)}, names=['zol'], headless=['zol']).names == ['zol']
    (team / 'zol' / 'promoted').write_text('bare\n')
    registry.save_checkpoint('zol', 'bare', scripted('WAIT 8'), root=team)
    with pytest.raises(RunnerError, match='replaces no network with a hit head'):
        Policies(team, {'zol': str(bare)}, names=['zol'], headless=['zol'])
    with pytest.raises(RunnerError, match="--headless \\['goriya'\\]: no --checkpoint"):
        Policies(team, {'zol': str(bare)}, names=['zol'], headless=['goriya'])


def _git(root: Path, *args):
    subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


def test_a_module_not_built_from_the_sources_is_refused(tmp_path):
    root = tmp_path / 'checkout'
    (root / 'crates' / 'a' / 'src').mkdir(parents=True)
    (root / 'triforce').mkdir()
    (root / 'crates' / 'a' / 'src' / 'lib.rs').write_text('fn a() {}\n')
    (root / 'Cargo.toml').write_text('[workspace]\n')
    _git(root, 'init', '-q')
    _git(root, 'add', '.')
    _git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'one')
    assert 'no built module' in build.problem(root)
    (root / 'triforce' / '_native.abi3.so').write_bytes(b'one')
    assert 'was not built by `triforce job build`' in build.problem(root)
    assert build.write_stamp(root) == 0 and build.problem(root) is None
    (root / 'triforce' / '_native.abi3.so').write_bytes(b'rebuilt by uv run')     # the same sources: no problem
    assert build.problem(root) is None
    (root / 'crates' / 'a' / 'src' / 'lib.rs').write_text('fn a() { 1; }\n')
    assert 'changed since the module was built' in build.problem(root)
    build.write_stamp(root)
    (root / 'crates' / 'a' / 'src' / 'new.rs').write_text('fn b() {}\n')          # an untracked source counts too
    assert 'changed since' in build.problem(root)


def test_a_stale_module_stops_a_tool_before_it_waits():
    result = subprocess.run([sys.executable, '-c', 'from triforce import jobs, build; '
                             'build.problem = lambda: "the crates changed"; jobs.announce("train")'],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2 and 'rebuild with `triforce job build' in result.stderr, result.stderr


def test_the_test_suites_read_the_main_checkouts_team(tmp_path, monkeypatch):
    monkeypatch.delenv('TRIFORCE_MODELS_DIR', raising=False)
    root = tmp_path / 'checkout'
    root.mkdir()
    _git(root, 'init', '-q')
    with pytest.raises(job.JobError, match='read the main checkout'):
        job.test_steps(root, ['python'], [])
    (root / 'models').mkdir()
    (step,) = job.test_steps(root, ['python'], [])
    assert step.env['TRIFORCE_MODELS_DIR'] == str(root / 'models')


def test_a_missing_demonstration_file_is_refused(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'triforce.train', '--specialist', 'ow-start', '--device', 'cpu',
                             '--start-state', 'start', '--demos', str(tmp_path / 'gone.npz'),
                             '--models-dir', str(tmp_path)], capture_output=True, text=True, check=False,
                            env={**os.environ, 'TRIFORCE_JOB_ESTIMATE': str(tmp_path / 'work.json')})
    assert result.returncode == 2 and f"--demos {tmp_path}/gone.npz: no such file" in result.stderr, result.stderr
