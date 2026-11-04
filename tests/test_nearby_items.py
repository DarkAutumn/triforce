"""Nearest helpful drop versus exit: actual shaping/pickups, not a heart-only objective."""

from triforce.actions import Action
from triforce.objectives import ObjectiveKind
from triforce.specialist import RewardConfig
from triforce.zelda_enums import ZeldaItemKind

from .test_env import greedy_step, make_env


def placed_env(session, drops, *, y=157, nearby=True):
    env = make_env(session, 'stalfos-keese-gel', '1_73e', one_room=True,
                   reward=RewardConfig(nearby_items=nearby))
    env.reset(seed=1)
    session.apply_ram({'link_x': 120, 'link_y': y, 'hearts_and_containers': 0x21, 'partial_hearts': 0xFF})
    session.emu.write(0x53, 0)       # placed off the doorway
    session.emu.write(0x394, 0)      # grid-aligned
    for slot, (kind, x, drop_y) in enumerate(drops, start=3):
        for address, value in ((0x34F, 0x60), (0xAC, kind.value), (0x70, x), (0x84, drop_y),
                               (0x405, 0), (0x3A8, 0xC0)):
            session.emu.write(address + slot, value)
    snapshot = session.snapshot()
    env.restore(snapshot, '1_73e')
    return env, snapshot


def movement_reward(env, direction):
    env.step(Action.parse(f'MOVE {direction} 4').to_indices())   # seeds critic's tile baseline
    _, _, _, _, info = env.step(Action.parse(f'MOVE {direction} 8').to_indices())
    rewards = info['rewards']
    return rewards.get('reward-pbrs-movement', 0) + rewards.get('penalty-pbrs-movement', 0)


def finish(env):
    for _ in range(100):
        _, _, terminated, truncated, info = greedy_step(env)
        if terminated or truncated:
            return info
    raise AssertionError('greedy nearest-target walker did not leave the room')


def test_nearest_clock_then_heart_then_exit_without_changing_task(session):
    env, snapshot = placed_env(session, [(ZeldaItemKind.Heart, 72, 157), (ZeldaItemKind.Clock, 152, 157)])
    assert movement_reward(env, 'E') > 0    # nearer clock wins over heart west and exit north
    assert env.session.state.objectives.kind == ObjectiveKind.MOVE
    info = finish(env)
    assert info['ending'] == 'exit-correct'
    assert info['hearts']['taken'] == 1 and info['hearts']['healed'] == 1
    assert info['health'] == 3

    legacy, _ = placed_env(session, [], nearby=False)
    legacy.restore(snapshot, '1_73e')
    assert movement_reward(legacy, 'E') < 0
    legacy.close()
    env.close()


def test_nearer_exit_leaves_distant_helpful_drops(session):
    env, _ = placed_env(session, [(ZeldaItemKind.Heart, 48, 189), (ZeldaItemKind.Clock, 200, 189)], y=125)
    assert movement_reward(env, 'N') > 0
    info = finish(env)
    assert info['ending'] == 'exit-correct'
    assert info['hearts']['taken'] == 0 and info['hearts']['left'] == 1
    assert info['health'] == 2
    env.close()
