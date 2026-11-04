"""Reward framework, terminal rewards, ending classification, and critic behavior on real savestates."""

from types import SimpleNamespace

import pytest

from triforce.actions import Action
from triforce.critics import GameplayCritic, REWARD_OPENED_LOCKED_DOOR, create_critic
from triforce.end_conditions import (DEFAULT_FAILURES, UnclassifiedEndingError, classify_ending, enter_room_name,
                                     parse_enter_room, validate_success_kind)
from triforce.projectile_blocks import ProjectileBlock
from triforce.rewards import (HARD_FAILURE, REWARD_MAXIMUM, REWARD_MEDIUM, SOFT_FAILURE, SUCCESS, Penalty,
                              Reward, StepRewards, apply_terminal_rewards, success_reward_value)
from triforce.specialist import RamRandomization, RewardConfig
from triforce.zelda_enums import ActionKind, Direction, MapLocation

from .test_env import chase_action, greedy_step, make_env
from .test_core import _place_shield_projectile


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



def _block_change(events=(), kind=ActionKind.WAIT):
    action = Action.parse('WAIT 4') if kind == ActionKind.WAIT else Action(kind, Direction.E)
    return SimpleNamespace(projectile_blocks=events, blocks=len(events), action=action,
                           previous=SimpleNamespace(link=SimpleNamespace(is_blocking=False)),
                           state=SimpleNamespace(link=SimpleNamespace(is_blocking=False)), health_lost=0)


@pytest.mark.parametrize('multiplier', [0.0, 1.0, 3.0])
def test_block_reward_multiplier_scales_unchanged_base_without_sound_onset(multiplier):
    event = ProjectileBlock(10, 11, 0x5C, MapLocation(1, 0x44, False))
    critic = create_critic('default', pbrs=False, block_reward=multiplier)
    rewards = StepRewards()
    critic.critique_block(_block_change((event,)), rewards)
    assert rewards.to_dict().get('reward-block-projectile', 0) == REWARD_MEDIUM * multiplier
    assert critic._room_combat_counts[event.full_location] == 1
    if multiplier:
        assert rewards['reward-block-projectile'].count == 1


def test_legacy_clink_without_collision_is_not_rewarded():
    critic = create_critic('default', pbrs=False)
    change = _block_change()
    change.state.link.is_blocking = True  # Darknut/immunity or a delayed old clink is not a Link collision
    rewards = StepRewards()
    critic.critique_block(change, rewards)
    assert rewards.to_dict() == {} and critic._room_combat_counts == {}


def test_each_block_advances_decay_in_its_collision_room_not_the_option_end_room():
    room, next_room = MapLocation(1, 0x44, False), MapLocation(1, 0x43, False)
    events = (ProjectileBlock(10, 10, 0x5C, room), ProjectileBlock(10, 11, 0x5C, room),
              ProjectileBlock(20, 11, 0x53, next_room))
    critic = GameplayCritic(pbrs=False, block_reward=2.0)
    critic._room_combat_counts[room] = 9  # next two decay factors are 0.5 and 0.25; new room remains 1.0
    rewards = StepRewards()
    change = _block_change(events)
    change.state.full_location = next_room
    critic.critique_block(change, rewards)
    outcome = rewards['reward-block-projectile']
    assert outcome.value == REWARD_MEDIUM * 2.0 * (0.5 + 0.25 + 1.0) and outcome.count == 3
    assert critic._room_combat_counts == {room: 11, next_room: 1}
    assert rewards.value == 1.0  # normal per-step clamping is unchanged


def test_grouping_blocks_into_longer_options_does_not_change_decay_or_total_reward():
    room = MapLocation(1, 0x44, False)
    events = tuple(ProjectileBlock(frame, 11, 0x5C, room) for frame in range(12))

    def total(groups):
        critic = GameplayCritic(pbrs=False)
        reward = 0.0
        count = 0
        for group in groups:
            rewards = StepRewards()
            critic.critique_block(_block_change(group), rewards)
            reward += rewards['reward-block-projectile'].value
            count += rewards['reward-block-projectile'].count
        return reward, count, critic._room_combat_counts

    assert total([events]) == total([(event,) for event in events])


def test_gameplay_rewards_a_confirmed_block_during_a_weapon_option(monkeypatch):
    critic = GameplayCritic(pbrs=False)
    # Other critiques are unrelated to deciding whether a real per-frame block belongs to an attack option.
    for name in ('critique_triforce', 'critique_attack', 'critique_item_usage', 'critique_equipment_pickup',
                 'critique_puzzles', 'critique_location_change', 'critique_movement',
                 'critique_health_change', 'critique_wallmaster'):
        monkeypatch.setattr(critic, name, lambda change, rewards: None)
    event = ProjectileBlock(10, 11, 0x5C, MapLocation(1, 0x44, False))
    rewards = StepRewards()
    critic.critique_gameplay(_block_change((event,), ActionKind.SWORD), rewards)
    assert rewards['reward-block-projectile'].value == REWARD_MEDIUM


def test_real_simultaneous_projectiles_each_earn_one_block_reward_and_not_again(session):
    _place_shield_projectile(session, slots=(10, 11))
    critic = GameplayCritic(pbrs=False)
    change = session.act(Action.parse('WAIT 16'))
    rewards = StepRewards()
    critic.critique_block(change, rewards)
    assert change.blocks == 2
    assert rewards['reward-block-projectile'].value == 2 * REWARD_MEDIUM
    assert rewards['reward-block-projectile'].count == 2
    follow = StepRewards()
    critic.critique_block(session.act(Action.parse('WAIT 4')), follow)
    assert follow.to_dict() == {}

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
    walks through into 0x63."""
    env = make_env(session, 'stalfos-keese-gel', '1_73e')
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
        env = make_env(session, 'stalfos-keese-gel', '1_73s_0', reward=RewardConfig(pbrs=pbrs))
        env.reset(seed=5)
        return [env.step(Action.parse(a).to_indices())[4]['rewards'] for a in script]

    on, off = run(True), run(False)
    pbrs_keys = ('reward-pbrs-movement', 'penalty-pbrs-movement')
    assert any(k in step for step in on for k in pbrs_keys)
    assert not any(k in step for step in off for k in pbrs_keys)
    strip = [{k: v for k, v in step.items() if k not in pbrs_keys} for step in on]
    assert strip == off


def test_wait_gets_no_movement_shaping(session):
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0')
    env.reset(seed=0)
    env.step(Action.parse('MOVE N 8').to_indices())          # seeds the PBRS baseline
    for _ in range(3):
        info = env.step(Action.parse('WAIT 16').to_indices())[4]
        assert not any('pbrs' in k for k in info['rewards'])
        assert 'penalty-attack-miss' not in info['rewards']


def test_pbrs_rewards_walking_toward_objective(session):
    """From the 0x73 S entry (no keys) the objective is W or E; walking toward an exit earns positive PBRS."""
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0')
    env.reset(seed=0)
    total = sum(info['rewards'].get('reward-pbrs-movement', 0.0) + info['rewards'].get('penalty-pbrs-movement', 0.0)
                for info in _walk(env, 8))
    assert total > 0


def test_sword_miss_is_penalized(session):
    env = make_env(session, 'goriya', '1_44w', ram_randomization=RamRandomization(hp=(0.5, 0.5)))  # sword, not beams
    env.reset(seed=0)
    for _ in range(10):         # off the door tile and the Goriyas out of their spawn clouds
        env.step(Action.parse('MOVE E 16').to_indices())
        if session.state.link.position.x >= 32 and session.state.active_enemies:
            break
    assert session.mask.kind[ActionKind.SWORD]
    info = env.step(Action.parse('SWORD S').to_indices())[4]   # goriyas are E of Link
    assert info['rewards'].get('penalty-attack-miss') == -0.01 or 'reward-hit' in info['rewards']


def test_damage_penalty_and_reward_halving(session, monkeypatch):
    """Taking damage costs 0.25 per half heart and halves the step's other positive rewards (not pickups)."""
    env = make_env(session, 'goriya', '1_44w', ram_randomization=RamRandomization(hp=(0.5, 0.5)))
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
