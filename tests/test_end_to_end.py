"""End to end over the whole stack (emulator, game model, envs, pools, Python CLI), each a few seconds.

Marked ``e2e`` (they run by default; ``-m e2e`` selects only them).  Networks are untrained: these check the
pipelines, not what a model learned."""

import json
import re

import pytest

from triforce import _native, chain_hp, cli, registry
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import observation_space
from triforce.runner import RunResult, chain_summary
from triforce.zelda_enums import ActionKind

from .test_demo import CAVE_ACTIONS

pytestmark = pytest.mark.e2e

ROOM_KEY = re.compile(r'^\d:0x[0-9a-f]{2}(:cave)?$')     # evaluate's by_room / chain_hp's profile keys
OUTSIDE, CAVE, NORTH = (0, 0x77, False), (0, 0x77, True), (0, 0x67, False)


def test_scripted_walk_takes_the_sword_and_leaves_the_start_screen(game):
    """From the game start: into the sword cave, take the wood sword, out again and north into 0x67, where the
    octoroks make the sword legal (beams at full health, the plain sword below)."""
    game.load('start', randomize_rng=False)
    assert game.state.sword == 0
    entries, pickup = [], None
    for step, action in enumerate(CAVE_ACTIONS, start=1):
        change = game.act(str(action))
        if change.changed_location:
            entries.append((step, change.previous.full_location, change.state.full_location))
        if change.state.sword and pickup is None:
            pickup = step, change.interrupt, change.state.full_location
    assert entries == [(17, OUTSIDE, CAVE), (30, CAVE, OUTSIDE), (len(CAVE_ACTIONS), OUTSIDE, NORTH)]
    assert pickup == (24, 'item', CAVE)
    state = game.state
    assert state.sword == 1 and state.has_beams and state.health == state.max_health == 3
    assert game.steps == len(CAVE_ACTIONS)

    def kinds():
        return {kind for kind in (ActionKind.SWORD, ActionKind.BEAMS) if game.mask[kind]}
    for _ in range(12):                                    # until the octoroks are active
        if kinds():
            break
        game.act('WAIT 8')
    assert kinds() == {ActionKind.BEAMS}
    game.set_value('hearts_and_containers', 0x21)          # 2 of 3 hearts: no beams
    game.act('WAIT 4')
    assert kinds() == {ActionKind.SWORD}


def test_evaluate_the_promoted_network_through_the_cli(states_dir, tmp_path, capsys):
    root = tmp_path / 'models'
    registry.save_checkpoint('ow-start', 'untrained', ZeldaPolicy(observation_space(), ModelConfig(hidden_dim=32)),
                             root=root)
    registry.promote('ow-start', 'untrained', root=root)
    config = tmp_path / 'ow-start.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 12\n")
    with pytest.raises(FileNotFoundError):
        cli.main(['evaluate', '--specialist', str(config), '--models-dir', str(root), '--run-id', 'missing'])

    assert cli.main(['evaluate', '--specialist', str(config), '--models-dir', str(root),
                     '--episodes-per-state', '3', '--parallel', '2', '--seed', '4']) == 0
    output = root / 'ow-start' / 'untrained' / 'eval.json'
    assert f"(written to {output})" in capsys.readouterr().out
    result = json.loads(output.read_text())
    records = result['records']
    assert (result['specialist'], result['episodes'], result['seed']) == ('ow-start', 3, 4)
    assert result['checkpoint'] == str(output.parent / 'ckpt.pt')
    assert [r['seed'] for r in records] == [4, 5, 6] and all(0 < r['steps'] <= 12 for r in records)
    assert set(result['by_state']) == {'start'} and result['by_state']['start']['episodes'] == 3
    assert all(ROOM_KEY.match(key) for key in result['by_room'])
    assert sum(room['episodes'] for room in result['by_room'].values()) == 3
    assert sum(result['outcomes'].values()) == sum(result['endings'].values()) == 3
    assert set(result['outcomes']) <= set(_native.OUTCOMES)
    assert result['mean_health_change'] == pytest.approx(
        sum(r['health'] - r['start_health'] for r in records) / 3)


def test_eval_chain_to_an_early_room_writes_runs_that_round_trip(models_dir, tmp_path, capsys):
    output = tmp_path / 'chain.jsonl'
    assert cli.main(['--models-dir', str(models_dir), 'eval-chain', '--runs', '3', '--parallel', '3',
                     '--stop-at', '0:0x67', '--output', str(output)]) == 0
    printed = capsys.readouterr().out
    lines = [json.loads(line) for line in output.read_text().splitlines()]
    results = [RunResult.from_dict(line) for line in lines]
    assert [r.to_dict() for r in results] == lines
    assert [r.seed for r in results] == [0, 1, 2]
    for result in results:
        first, last = result.rooms[0], result.rooms[-1]
        assert (first.level, first.room, first.in_cave, first.specialist) == (0, '0x77', False, 'ow-start')
        assert [room.index for room in result.rooms] == list(range(len(result.rooms)))
        assert result.steps == sum(room.steps for room in result.rooms)
        assert result.frames == sum(room.frames for room in result.rooms)
        assert first.health_in == first.max_health == 3
        if result.result == 'success':
            assert (last.level, last.room, last.in_cave) == (0, '0x67', False)
            assert result.reason == 'entered (level 0, room 0x67)'
        else:
            assert result.result == 'failure' and last.outcome != 'in-progress'
    summary = json.loads(printed[:printed.rindex('}') + 1])
    assert summary == chain_summary(results)

    # The chain jsonl feeds the chain-hp profile; its keys are the native lookup's room keys.
    profile = chain_hp.build_profile([str(output)])
    assert all(ROOM_KEY.match(key) for key in profile)
    assert sum(map(len, profile.values())) == sum(len(r.rooms) for r in results)
