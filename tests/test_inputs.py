"""Round 40 (the owner: missing is an error, never "none"): every input a run declares that resolves to nothing stops
it before it starts, naming where it looked; where none is a legitimate recipe the task declares it."""

import fnmatch
import hashlib
import json
import os
import re
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
from triforce.runner import Policies, Route, RunnerError, check_configs, check_team, switched_reads
from triforce.specialist import SpecialistConfig, load_specialist, task_files

from .conftest import REPO_STATES_DIR, require_rom
from .networks import scripted

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def zol():
    """zol@d3first's task (ten start-state globs, a curriculum) with harvested states at 0.25 (the generation draws
    none since round 43, the reset rule; the machinery stands for a task that sets a share)."""
    require_rom()
    return load_specialist(str(ROOT / 'specialists' / 'd3first' / 'zol.yaml')).replace(harvested_fraction=0.25)


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
    tracked = _tracks(tmp_path / 'tracks', seeds[:-1])                  # round 44: one state per door
    assert set(track_starts(tracked).values()) == set(seeds[:-1])
    bare = zol.replace(curriculum={'enabled': True})
    with pytest.raises(MissingInput, match=r"1 of \d+ seed states have no curriculum track") as refused:
        check_curriculum(bare, tracked, REPO_STATES_DIR)
    glob = seeds[-1][:-1] + '[0-9]'
    assert f"curriculum.untracked: ['{glob}']" in str(refused.value)
    declared = zol.replace(curriculum={'enabled': True, 'untracked': [glob]})
    assert len(check_curriculum(declared, tracked, REPO_STATES_DIR)) == len(seeds) - 1


def test_a_declaration_never_covers_a_tracked_state():
    seeds = ['a_0', 'a_1', 'a_2', 'b_0', 'b_1']
    assert declaration(['a_1', 'b_0', 'b_1'], seeds) == ['a_1', 'b_[0-9]']


def test_the_shipped_tasks_declare_every_input_they_draw():
    """Every task of specialists/ passes its checks on this checkout's data (the declarations of round 40)."""
    require_rom()
    for path in task_files():
        config = load_specialist(str(path))
        trajectories = (str(config.curriculum.trajectories or paths.data_dir('trajectories') / config.name)
                        if config.curriculum.enabled else None)
        if trajectories is not None and not Path(trajectories).is_dir():
            pytest.skip(f"{trajectories}: this checkout has no trajectories")
        check_task(config, REPO_STATES_DIR, trajectories)


# Round 40 (Tracks40): the seed states whose stale tracks left the generation's curricula and that the explorer did
# not solve in 3000 iterations (with darknut's 0x59 from N at 4 bombs, States40's), per family, beside the same
# rooms' chains that do have a track. Both of the family's tasks draw them (its d3first part's and its own).
# Tracks41 gave all 25 a track (round40.md "The untracked states' tracks": the scripted rules, Darknut40's fighter,
# the explorer at a larger budget), so none is left and every one of them is a chain with a track now. Round 44:
# d3first has one state per door (chain 0), so each room's seed is its `_0` state alone.
LEFT_WITHOUT_A_TRACK = {
    'darknut': (set(), {f'd3first_3_{room}_0' for room in ('59n', '5bw')}),
    'zol': (set(), {'d3first_3_4bs_0'}),
    'vire': (set(), {f'd3first_4_{room}_0' for room in ('12n', '31w')}),
    'like-like': (set(), {'d3first_4_32w_0', 'd3first_4_32c_0'}),
    'boss-gleeok': (set(), {'d3first_4_13w_0'}),
    'goriya': (set(), {f'd3first_{room}_0' for room in ('1_23s', '2_1es', '7_1bw')}),
    'boss-digdogger': (set(), {'d3first_7_1cw_0', 'd3first_7_39s_0', 'd3first_5_24e_0'}),
}


def test_the_d3first_tasks_declare_the_rooms_left_without_a_track():
    """Round 40 (the lead, Nov 24): 0:0x76's explored tracks walk across and kill, so ow-tektite@d3first trains its
    seed (round 44: one state per door, chain 0) without one; and the states left without a track
    (`LEFT_WITHOUT_A_TRACK`: Tracks40's, all tracked since Tracks41). The tasks load, and the declarations cover
    exactly those seeds: never a chain of the same room that has a track."""
    require_rom()
    tektite = load_specialist(str(ROOT / 'specialists' / 'd3first' / 'ow-tektite.yaml'))
    seeds = sorted(Path(p).stem for p in tektite.resolve_start_states(str(REPO_STATES_DIR)))
    assert [s for s in seeds if s.startswith('d3first_0_76e_')] == ['d3first_0_76e_0']
    assert declaration(['d3first_0_76e_0'], seeds) == untracked(tektite)
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


# Round 45 (the lead, Nov 26): the walked d3first doors a route step's specialist task deliberately does not draw as a
# seed state, with the reason. Empty: ow-lynel@d3first's Death Mountain (0x05-0x32), boss-gleeok@d3first's 8:0x3c and
# ow-tektite@d3first's 0x1e and 0x0d were left out of their `start_states` (round 38 and the rebuilt run tasks) and
# nothing flagged it; they are seeds now.
NOT_DRAWN: dict[str, str] = {}


def test_every_walked_door_is_a_seed_of_its_steps_task():
    """Every walked door of d3first (`triforce/states/d3first_walk.json`) is a seed state of its route step's
    specialist's d3first task (`specialists/d3first/<family>.yaml`), so a reset can start the room the route plays
    there; a door the task deliberately leaves to the chains is listed in `NOT_DRAWN` with its reason."""
    require_rom()
    from triforce import entry_states   # pylint: disable=import-outside-toplevel
    steps = {step.index: step for step in entry_states.route_steps('d3first')}
    doors = json.loads((REPO_STATES_DIR / 'd3first_walk.json').read_text(encoding='utf-8'))['doors']
    seeds = {}
    missing = []
    for name, record in doors.items():
        stem = Path(name).stem
        family = steps[record['route_step']].specialist
        if family not in seeds:
            config = load_specialist(str(ROOT / 'specialists' / 'd3first' / f'{family}.yaml'))
            seeds[family] = {Path(p).stem for p in config.resolve_start_states(str(REPO_STATES_DIR))}
        if stem not in seeds[family] and stem not in NOT_DRAWN:
            missing.append(f'{family}: {stem}')
    assert not missing, missing
    assert not set(NOT_DRAWN) - {Path(name).stem for name in doors}, 'NOT_DRAWN names a door the walk lacks'


# Round 40, the fresh generation (2.0.1.0): every teacher's task, `specialists/d3first/<family>.yaml` (the lead's pins:
# four groups and seven bosses, every family of the route).
GENERATION = ['ow-start', 'ow-tektite', 'ow-leever', 'ow-moblin', 'ow-octorok', 'ow-zora', 'ow-peahat', 'ow-lynel',
              'ow-ghini', 'ow-boulder', 'darknut', 'gibdo', 'stalfos-keese-gel', 'zol', 'like-like', 'vire', 'goriya',
              'd2-rope', 'dodongo-crossing', 'moldorm', 'wizzrobe', 'pols-voice', 'wallmaster', 'lanmola', 'patra',
              'equipment-room', 'trap-room', 'boss-manhandla', 'boss-gleeok', 'boss-aquamentus', 'boss-dodongo',
              'boss-digdogger', 'boss-gohma', 'ganon']

# The effective configs (SHA-256 of `to_dict()`, as `manifest.task` hashes them; first 16 hex) of the bare task
# files, the ones today's team and every `d2first` chain play with: unchanged by the generation's tasks (r40-tasks:
# the same before and after on `r40-gen` 42d15e02). A deliberate change to a bare task updates its row: Tracks41
# (r40-gen fee9db0c) dropped `curriculum.untracked` entries from boss-digdogger, boss-gleeok, darknut, goriya,
# like-like, vire and zol (their states got tracks); Tracks45 (round 45) dropped the rest of them (every walked
# d3first door got tracks): boss-dodongo, boss-gleeok, boss-gohma, boss-manhandla, d2-rope, darknut, dodongo-crossing,
# equipment-room, ganon, gibdo, goriya, lanmola, like-like, moldorm, ow-boulder, ow-ghini, ow-lynel, ow-moblin,
# ow-peahat, ow-start, ow-zora, patra, pols-voice, trap-room, vire, wizzrobe and zol.
BARE_EFFECTIVE = {
    'boss-aquamentus': '964037cc7c6772f7', 'boss-digdogger': '9fc25620d8af66a8', 'boss-dodongo': '7461eb48c333d15a',
    'boss-gleeok': '192da91fd33fe237', 'boss-gohma': '4dbb47a3c834ff73', 'boss-manhandla': '8743ae0fc9027b57',
    'd2-rope': '279e4949c7a2486f', 'darknut': '0954c38e71ae04a5', 'dodongo-crossing': 'c76e737302687a38',
    'equipment-room': '9104ac87057c87a9', 'ganon': 'a9a0946f210b208f', 'gibdo': '73fa4086f6f1bb6f',
    'goriya': 'a24a98a6ca911d59', 'lanmola': '8b26e87dc23965cd', 'like-like': '5456bb8f8dcabd4e',
    'moldorm': '722cf1df9adcfa1f', 'ow-boulder': 'f1e62898c1a86a81', 'ow-ghini': 'a9f2595f23135885',
    'ow-leever': '4d74f994b3d7d936', 'ow-lynel': 'b494620d0ef3ab8e', 'ow-moblin': '44379ed012f710b9',
    'ow-octorok': '796292f6ddc66d68', 'ow-peahat': 'cf49a64afa791bfe', 'ow-start': 'e37b0a27c4a124dd',
    'ow-tektite': '729c50d784b9d954', 'ow-zora': '3a0353c57ebbb36a', 'patra': '64baa281838a7c56',
    'pols-voice': '6be5d292dfa6538c', 'stalfos-keese-gel': '8039e87cffe1a7ef', 'trap-room': '3baa0e14cfe200be',
    'vire': 'f6ab857fd54ef5c6', 'wallmaster': 'feca0df6d2aac23f', 'wizzrobe': 'd0c8d9b6f762e71a',
    'zol': '04d329a56258c10c',
}


def _effective(config) -> str:
    return hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def test_every_generation_task_trains_from_scratch_with_the_generations_inputs():
    """Round 40 (the lead's pins): every teacher's d3first task has `obs.threat` and `obs.chase` on, the aimless
    penalty at 0.002 (`_generation.yaml`, one place), no chase penalty (an arm, not a default), no warm start, and
    the route's states only; the exposed-attack penalty at 0.1 in the Goriya teacher's task and the walkers' eight
    (the owner, Nov 26), nowhere else; `obs.shooter` in the Goriya teacher's task, the overworld teacher's ten
    ("Every shooter") and the Wizzrobe teacher's (Nov 27), nowhere else; darknut's bombs catch two or more; the Bubble
    rooms' items as Switch40 measured."""
    exposed = {'goriya', 'gibdo', 'stalfos-keese-gel', 'zol', 'like-like', 'vire', 'd2-rope', 'dodongo-crossing',
               'moldorm'}
    shooter = {'goriya', 'wizzrobe', 'ow-start', 'ow-tektite', 'ow-leever', 'ow-moblin', 'ow-octorok', 'ow-zora',
               'ow-peahat', 'ow-lynel', 'ow-ghini', 'ow-boulder'}
    require_rom()
    tasks = sorted(p.stem for p in task_files(ROOT / 'specialists' / 'd3first'))
    assert tasks == sorted(GENERATION) == sorted(SpecialistConfig.names())
    for family in GENERATION:
        config = load_specialist(str(ROOT / 'specialists' / 'd3first' / f'{family}.yaml'))
        assert config.name == family
        assert config.obs.threat and config.obs.chase, family
        assert config.to_dict()['obs'].get('shooter', False) == (family in shooter), family
        assert config.reward.aimless_penalty == 0.002 and config.reward.chase_penalty == 0, family
        assert config.to_dict()['reward'].get('exposed_attack_penalty', 0) == (0.1 if family in exposed else 0), family
        assert config.warm_start is None, family
        assert list(config.state_routes) == ['d3first'], family
    darknut = load_specialist(str(ROOT / 'specialists' / 'd3first' / 'darknut.yaml'))
    assert darknut.items.bomb_catch == 2 and darknut.items.bombs == 'free'
    wallmaster = load_specialist(str(ROOT / 'specialists' / 'd3first' / 'wallmaster.yaml'))   # 7:0x0d
    assert wallmaster.items.boomerang and wallmaster.items.candle
    wizzrobe = load_specialist(str(ROOT / 'specialists' / 'd3first' / 'wizzrobe.yaml'))       # 6:0x38
    assert wizzrobe.items.boomerang and not wizzrobe.items.candle


def test_every_generation_task_draws_the_drop_state_by_one_rule_and_the_bosses_keep_their_bombs():
    """Round 43 (the owner, Nov 25): one rule in `_generation.yaml` for every teacher's task, no task file with a
    counter draw of its own: the forced-drop counters as real play holds them at a door (`drop_state: real`) and the
    `help_count` streak only where a room states its own drop cycle (darknut's 3:0x5b from S, 0 or 1, as before).
    Manhandla's and Dodongo's tasks end an exit that leaves their bombs lying (`drops-left`), point at the drops, and
    arm the bomb drop in training (0.75 and 0.5 of the resets); no other task names the ending."""
    require_rom()
    folder = ROOT / 'specialists' / 'd3first'
    for family in GENERATION:
        text = (folder / f'{family}.yaml').read_text(encoding='utf-8')
        assert not re.search(r'^\s*(help_count|drop_state)\s*:|[{,]\s*help_count\s*:', text, re.M), family
        rr = load_specialist(str(folder / f'{family}.yaml')).to_dict()['ram_randomization']
        assert rr['drop_state'] == 'real' and list(rr['help_count']) == [0, 9], family
        endings = load_specialist(str(folder / f'{family}.yaml')).to_dict()['endings']
        assert ('drops-left' in endings) == (family in ('boss-manhandla', 'boss-dodongo')), family
    darknut = load_specialist(str(folder / 'darknut.yaml')).to_dict()['ram_randomization']
    assert {k: list(v) for k, v in darknut['room_kill_cycle'].items()} == {'(3, 0x5b, S)': [0, 1]}
    for family, share in (('boss-manhandla', 0.75), ('boss-dodongo', 0.5)):
        task = load_specialist(str(folder / f'{family}.yaml')).to_dict()
        assert task['endings']['drops-left'] == 'soft-failure', family
        assert task['reward']['drops_first'] and task['ram_randomization']['drop_bombs'] == share, family


def test_the_generations_tasks_change_no_bare_task_and_no_d2first_chain():
    """The bare task files (today's team, `d2first`) are as `r40-gen` had them, and a `d2first` chain plays every
    family with its bare task; the common block is no family."""
    require_rom()
    assert {family: _effective(load_specialist(family)) for family in GENERATION} == BARE_EFFECTIVE
    for family in GENERATION:
        assert SpecialistConfig.for_chain(family, 'd2first').to_json() == load_specialist(family).to_json(), family
    assert '_generation' not in SpecialistConfig.names()


def test_a_d3first_chain_takes_the_route_tasks_observation_switches_and_nothing_else():
    """`SpecialistConfig.for_chain` (`GameData::chain_specialist`): the family's own task with the route task's `obs`
    block (`CHAIN_FROM_ROUTE_TASK`); items, rewards and every other rule stay the bare task's."""
    require_rom()
    for family in GENERATION:
        chained = json.loads(SpecialistConfig.for_chain(family, 'd3first').to_json())
        bare = json.loads(load_specialist(family).to_json())
        routed = load_specialist(str(ROOT / 'specialists' / 'd3first' / f'{family}.yaml'))
        assert chained['obs'] == json.loads(routed.to_json())['obs'], family
        assert {k: v for k, v in chained.items() if k != 'obs'} == {k: v for k, v in bare.items() if k != 'obs'}
    darknut = SpecialistConfig.for_chain('darknut', 'd3first')
    assert darknut.obs.threat and darknut.obs.chase
    assert darknut.items.bomb_catch == 0 and darknut.reward.aimless_penalty == 0    # the bare task's rules


def _reader(tokens):
    from triforce import nets
    from .networks import native_spec
    small = {'d_model': 32, 'layers': 1, 'link_out': 32, 'hidden_dim': 32, 'depth': 1, 'hit_dim': 32, 'hit_depth': 1}
    return nets.create('hitnet', native_spec(), tokens=tokens, feed=True, **small)


def test_a_chain_refuses_a_network_that_reads_a_block_its_switch_leaves_zero(tmp_path):
    """A `hitnet` reading `entity_threat` / `chase` is refused where the config it is chained with has the switch
    off (every `d2first` chain today), naming the family and the switch; on `d3first` (the generation's tasks) it
    plays; a network without those groups plays anywhere."""
    require_rom()
    reader = _reader(['type', 'features', 'rel', 'threat', 'chase'])
    assert switched_reads(reader) == {'entity_threat': 'threat', 'entity_chase': 'chase', 'chase': 'chase'}
    with pytest.raises(RunnerError) as refused:
        Policies(tmp_path, networks={'darknut': reader}, names=['darknut'], route='d2first')
    message = str(refused.value)
    assert "family darknut's network reads" in message and 'obs.chase and obs.threat are off' in message
    assert 'specialists/darknut.yaml (route d2first has no task of its own for it)' in message
    assert Policies(tmp_path, networks={'darknut': reader}, names=['darknut'], route='d3first').names
    threat_only = _reader(['type', 'features', 'rel', 'threat'])
    threat_on = load_specialist('darknut').replace(obs={'objective': True, 'target': True, 'threat': True})
    with pytest.raises(RunnerError, match=r"reads entity_threat \(obs\.threat\), but obs\.threat is off"):
        Policies(tmp_path, networks={'darknut': threat_only}, names=['darknut'], route='d2first')
    assert Policies(tmp_path, networks={'darknut': threat_only}, specialist_configs={'darknut': threat_on},
                    names=['darknut'], route='d2first').names == ['darknut']
    plain = _reader(['type', 'features', 'rel', 'contact'])
    assert switched_reads(plain) == {}
    assert Policies(tmp_path, networks={'darknut': plain}, names=['darknut'], route='d2first').names == ['darknut']


# `env::config::TRAINED_FROM_ROUTE_TASK`: the keys a slot whose network trained on the route's task takes from it.
TRAINED_KEYS = ('obs', 'items', 'clips', 'reward', 'max_steps', 'no_progress_steps', 'one_room')


def test_a_slot_trained_on_the_route_task_takes_every_rule_the_chain_reads_from_it():
    """`for_chain(..., trained=True)`: the route task's `TRAINED_KEYS` over the family's own task, nothing else
    (`rooms`, `endings`, `start_states`... stay the bare task's: no chain reads them for play)."""
    require_rom()
    for family in GENERATION:
        trained = json.loads(SpecialistConfig.for_chain(family, 'd3first', trained=True).to_json())
        bare = json.loads(load_specialist(family).to_json())
        routed = json.loads(load_specialist(str(ROOT / 'specialists' / 'd3first' / f'{family}.yaml')).to_json())
        assert {k: trained[k] for k in TRAINED_KEYS} == {k: routed[k] for k in TRAINED_KEYS}, family
        assert ({k: v for k, v in trained.items() if k not in TRAINED_KEYS}
                == {k: v for k, v in bare.items() if k not in TRAINED_KEYS}), family
    darknut = SpecialistConfig.for_chain('darknut', 'd3first', trained=True)
    assert darknut.items.bomb_catch == 2 and darknut.reward.aimless_penalty == 0.002
    assert SpecialistConfig.for_chain('wallmaster', 'd3first', trained=True).items.candle
    # d2first has no tasks of its own: a trained slot there would get the bare task (and the guard refuses it).
    bare_darknut = load_specialist('darknut').to_json()
    assert SpecialistConfig.for_chain('darknut', 'd2first', trained=True).to_json() == bare_darknut


def test_each_slot_plays_with_the_rules_its_network_trained_under(tmp_path):
    """Round 40 (the lead): per slot, by the network in it. A generation network (it reads the switched blocks) plays
    the route task's masks; today's network in the other slot of the same family keeps the bare masks."""
    require_rom()
    reader = _reader(['type', 'features', 'rel', 'threat', 'chase'])
    policies = Policies(tmp_path, networks={'darknut': scripted('WAIT 8'), 'darknut@d3first': reader,
                                            'wallmaster': reader}, names=['darknut', 'wallmaster'], route='d3first')
    configs = dict(zip(policies.names, policies.configs))
    assert configs['darknut@d3first'].items.bomb_catch == 2 and configs['darknut'].items.bomb_catch == 0
    assert configs['darknut'].obs.threat and configs['darknut'].reward.aimless_penalty == 0   # obs only
    assert configs['wallmaster'].items.candle
    today = Policies(tmp_path, networks={'wallmaster': scripted('WAIT 8')}, names=['wallmaster'], route='d3first')
    assert not today.configs[0].items.candle


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
    # zol@d3first's task plus the one line (round 40: the task extends `_generation.yaml` beside it, so a copy of its
    # text elsewhere no longer loads; the test's task extends it instead, as an experiment's file does).
    task = tmp_path / 'zol.yaml'
    task.write_text(f"extends: {ROOT / 'specialists' / 'd3first' / 'zol.yaml'}\nharvested_fraction: 0.25\n"
                    f"harvested_states: {tmp_path / 'none'}\n")
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


def test_the_test_suites_play_the_stores_current_team(tmp_path, monkeypatch):
    monkeypatch.delenv('TRIFORCE_MODELS_DIR', raising=False)
    monkeypatch.delenv('TRIFORCE_TEAM', raising=False)
    shared = tmp_path / 'shared'
    monkeypatch.setenv('TRIFORCE_SHARED', str(shared))
    root = tmp_path / 'checkout'
    (root / 'models').mkdir(parents=True)          # a checkout's own folder is never the suites' team
    with pytest.raises(job.JobError, match="play the store's current team and there is none"):
        job.test_steps(root, ['python'], [])
    (shared / 'teams').mkdir(parents=True)
    (shared / 'store.db').write_bytes(b'')
    (shared / 'teams' / 'current').write_text('2.0.0.15\n')
    (step,) = job.test_steps(root, ['python'], [])
    assert step.env['TRIFORCE_TEAM'] == '2.0.0.15' and 'TRIFORCE_MODELS_DIR' not in step.env
    monkeypatch.setenv('TRIFORCE_TEAM', '2.0.0.14')      # the caller's team is kept
    (step,) = job.test_steps(root, ['python'], [])
    assert 'TRIFORCE_TEAM' not in step.env


def test_a_missing_demonstration_file_is_refused(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'triforce.train', '--specialist', 'ow-start', '--device', 'cpu',
                             '--start-state', 'start', '--demos', str(tmp_path / 'gone.npz'),
                             '--models-dir', str(tmp_path)], capture_output=True, text=True, check=False,
                            env={**os.environ, 'TRIFORCE_JOB_ESTIMATE': str(tmp_path / 'work.json')})
    assert result.returncode == 2 and f"--demos {tmp_path}/gone.npz: no such file" in result.stderr, result.stderr
