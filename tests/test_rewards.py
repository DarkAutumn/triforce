"""Reward framework, terminal rewards, ending classification, and critic behavior on real savestates."""

import pytest

from triforce.actions import Action
from triforce.critics import REWARD_OPENED_LOCKED_DOOR
from triforce.end_conditions import (DEFAULT_FAILURES, UnclassifiedEndingError, classify_ending, enter_room_name,
                                     parse_enter_room, validate_success_kind)
from triforce.rewards import (HARD_FAILURE, REWARD_MAXIMUM, SOFT_FAILURE, SUCCESS, Penalty, Reward, StepRewards,
                              apply_terminal_rewards, success_reward_value)
from triforce.specialist import RamRandomization, RewardConfig
from triforce.zelda_enums import MapLocation

from .test_env import greedy_step, make_env


# --- StepRewards ---

def test_step_value_is_clamped_to_one():
    rewards = StepRewards()
    rewards.add(Reward('reward-a', 0.8))
    rewards.add(Reward('reward-b', 0.8))
    assert rewards.value == REWARD_MAXIMUM
    rewards = StepRewards()
    rewards.add(Penalty('penalty-a', -0.9))
    rewards.add(Penalty('penalty-b', -0.9))
    assert rewards.value == -REWARD_MAXIMUM


def test_duplicate_outcome_is_rejected():
    rewards = StepRewards()
    rewards.add(Reward('reward-a', 0.1))
    with pytest.raises(AssertionError):
        rewards.add(Reward('reward-a', 0.1))


def test_health_halving_exempts_equipment():
    rewards = StepRewards()
    rewards.add(Reward('reward-hit', 0.5))
    rewards.add(Reward('reward-gained-keys', 1.0))
    rewards.add(Penalty('penalty-lost-health', -0.25))
    rewards.scale_rewards(0.5, exempt_prefixes=('reward-gained-',))
    assert rewards.to_dict() == {'reward-hit': 0.25, 'reward-gained-keys': 1.0, 'penalty-lost-health': -0.25}


@pytest.mark.parametrize('health,max_health,expected', [(3, 3, 20.0), (1.5, 3, 15.0), (0.5, 16, 10.3125),
                                                         (0, 3, 10.0)])
def test_success_reward_scales_with_health(health, max_health, expected):
    assert success_reward_value(health, max_health) == pytest.approx(expected)


def test_success_terminal_escapes_step_clamp():
    rewards = StepRewards()
    rewards.add(Reward('reward-new-location', 1.0))
    apply_terminal_rewards(rewards, 'exit-correct', SUCCESS, True, 3, 3)
    assert rewards.value == 20.0                       # 21 clamped to the terminal maximum
    assert rewards.ending == 'exit-correct' and rewards.outcome == SUCCESS


def test_hard_failure_drops_rewards_soft_keeps_them():
    hard = StepRewards()
    hard.add(Reward('reward-hit', 0.25))
    hard.add(Penalty('penalty-lost-health', -0.5))
    apply_terminal_rewards(hard, 'death', HARD_FAILURE, True, 0, 3)
    assert hard.to_dict() == {'penalty-lost-health': -0.5, 'penalty-terminal-failure': -20.0}
    assert hard.value == -20.0                          # -20.5 clamped to the terminal minimum

    soft = StepRewards()
    soft.add(Reward('reward-hit', 0.25))
    apply_terminal_rewards(soft, 'exit-wrong', SOFT_FAILURE, True, 3, 3)
    assert soft.value == pytest.approx(-19.75)


def test_truncation_records_but_adds_no_terminal():
    rewards = StepRewards()
    rewards.add(Penalty('penalty-room-stalling', -0.01))
    apply_terminal_rewards(rewards, 'max-steps', SOFT_FAILURE, False, 3, 3)
    assert rewards.outcome == SOFT_FAILURE and rewards.value == pytest.approx(-0.01)


# --- classification ---

def test_classification_is_deny_by_default():
    assert classify_ending('exit-correct', ['exit-correct']) == SUCCESS
    assert classify_ending('death', ['exit-correct']) == HARD_FAILURE
    assert classify_ending('exit-wrong', ['exit-correct']) == SOFT_FAILURE
    with pytest.raises(UnclassifiedEndingError):
        classify_ending('triforce-collected', ['exit-correct'])     # a success kind the specialist didn't list
    with pytest.raises(UnclassifiedEndingError):
        classify_ending('mystery', ['exit-correct'])
    assert classify_ending('exit-wrong', ['exit-correct'], {'exit-wrong': HARD_FAILURE}) == HARD_FAILURE
    assert not set(DEFAULT_FAILURES) & {'exit-correct', 'triforce-collected', 'boss-dead-and-heart'}


def test_enter_room_kinds_round_trip():
    assert parse_enter_room('enter-room:2:0x0e') == MapLocation(2, 0x0e, False)
    assert enter_room_name(MapLocation(2, 0x0e, False)) == 'enter-room:2:0x0e'
    assert validate_success_kind('enter-room:2:0xE') == 'enter-room:2:0x0e'
    assert enter_room_name(parse_enter_room('enter-room:0:0x77:cave')) == 'enter-room:0:0x77:cave'
    for bad in ('exit-sideways', 'enter-room:2', 'enter-room:x:0x0e'):
        with pytest.raises(ValueError):
            validate_success_kind(bad)


# --- critics on the emulator ---

def _walk(env, steps):
    infos = []
    for _ in range(steps):
        _, _, terminated, truncated, info = greedy_step(env)
        infos.append(info)
        if terminated or truncated:
            break
    return infos


def test_locked_door_reward_fires_once(session):
    """1_73e has 2 keys and the N door is locked: walking to it opens the door, rewarded once, then Link
    walks through (a correct exit inside d1-melee, so the episode continues)."""
    env = make_env(session, 'd1-melee', '1_73e')
    env.reset(seed=0)
    infos = _walk(env, 40)
    hits = [i for i, info in enumerate(infos) if REWARD_OPENED_LOCKED_DOOR.name in info['rewards']]
    assert len(hits) == 1
    assert infos[hits[0]]['rewards'][REWARD_OPENED_LOCKED_DOOR.name] == REWARD_OPENED_LOCKED_DOOR.value
    assert any(info['room'] == (1, 0x63, False) for info in infos[hits[0]:])


def test_pbrs_switch_changes_only_movement_shaping(session):
    """Same start, same RNG, same actions: PBRS on adds pbrs-movement outcomes, PBRS off has none, and every
    other outcome is identical."""
    script = ['MOVE N 16', 'MOVE N 16', 'MOVE W 16', 'MOVE S 8', 'MOVE W 16', 'WAIT 8']

    def run(pbrs):
        env = make_env(session, 'd1-melee', '1_73n', reward=RewardConfig(pbrs=pbrs))
        env.reset(seed=5)
        return [env.step(Action.parse(a).to_indices())[4]['rewards'] for a in script]

    on, off = run(True), run(False)
    pbrs_keys = ('reward-pbrs-movement', 'penalty-pbrs-movement')
    assert any(k in step for step in on for k in pbrs_keys)
    assert not any(k in step for step in off for k in pbrs_keys)
    strip = [{k: v for k, v in step.items() if k not in pbrs_keys} for step in on]
    assert strip == off


def test_wait_gets_no_movement_shaping(session):
    env = make_env(session, 'd1-melee', '1_73n')
    env.reset(seed=0)
    env.step(Action.parse('MOVE N 8').to_indices())          # seeds the PBRS baseline
    for _ in range(3):
        info = env.step(Action.parse('WAIT 16').to_indices())[4]
        assert not any('pbrs' in k for k in info['rewards'])
        assert 'penalty-attack-miss' not in info['rewards']


def test_pbrs_rewards_walking_toward_objective(session):
    """From the 0x73 S entry (no keys) the objective is W or E; walking toward an exit earns positive PBRS."""
    env = make_env(session, 'd1-melee', '1_73n')
    env.reset(seed=0)
    total = sum(info['rewards'].get('reward-pbrs-movement', 0.0) + info['rewards'].get('penalty-pbrs-movement', 0.0)
                for info in _walk(env, 8))
    assert total > 0


def test_sword_miss_is_penalized(session):
    env = make_env(session, 'd1-wallmaster', '1_44w')
    env.reset(seed=0)
    env.step(Action.parse('MOVE E 16').to_indices())
    info = env.step(Action.parse('SWORD S').to_indices())[4]   # goriyas are E of Link
    assert info['rewards'].get('penalty-attack-miss') == -0.01 or 'reward-hit' in info['rewards']


def chase_action(state):
    """MOVE 8 toward the nearest active enemy along the larger axis (to get hit)."""
    enemy = min(state.active_enemies, key=lambda e: e.distance, default=None)
    if enemy is None:
        return 'WAIT 8'
    dx, dy = enemy.position.x - state.link.position.x, enemy.position.y - state.link.position.y
    if abs(dx) > abs(dy):
        return 'MOVE E 8' if dx > 0 else 'MOVE W 8'
    return 'MOVE S 8' if dy > 0 else 'MOVE N 8'


def test_damage_penalty_and_reward_halving(session, monkeypatch):
    """Taking damage costs 0.25 per half heart and halves the step's other positive rewards (not pickups)."""
    env = make_env(session, 'd1-wallmaster', '1_44w', ram_randomization=RamRandomization(hp=(0.5, 0.5)))
    env.reset(seed=1)
    monkeypatch.setattr(env.critic, 'critique_triforce', lambda change, rewards: rewards.add(Reward('reward-hit', 0.5)))
    env.step(Action.parse('MOVE E 16').to_indices())
    for i in range(120):
        before = session.state.link.health
        action = chase_action(session.state)
        if not session.mask.is_allowed(Action.parse(action)):
            action = 'WAIT 8'
        _, _, terminated, _, info = env.step(Action.parse(action).to_indices())
        lost = before - session.state.link.health
        if lost > 0 and not terminated:
            assert info['rewards']['penalty-lost-health'] == pytest.approx(-min(0.25 * lost * 2, 1.0))
            assert info['rewards']['reward-hit'] == pytest.approx(0.25)
            return
        assert info['rewards'].get('reward-hit') == pytest.approx(0.5)
        if terminated:
            break
    pytest.skip("Link never took non-fatal damage")


def test_stalling_penalty_fires_in_zora_room(session):
    """The Zora surfaces and submerges every few steps; that must not reset the room-stalling counter (only kills
    and pickups do), so dithering at the 0x38 entrance is penalized after the grace period."""
    env = make_env(session, 'ow-zora', '0_38s')
    env.reset(seed=0)
    script = ['WAIT 8', 'WAIT 8', 'MOVE E 4', 'MOVE W 4']
    for i in range(300):
        _, _, terminated, truncated, info = env.step(Action.parse(script[i % len(script)]).to_indices())
        assert not (terminated or truncated), info['ending']
        if 'penalty-room-stalling' in info['rewards']:
            return
    pytest.fail("no room-stalling penalty in 300 steps of dithering")
