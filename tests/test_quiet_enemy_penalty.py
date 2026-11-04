"""A stronger quiet-fight cost uses real shared progress and stops when the fight is over."""

from triforce.actions import Action
from triforce.critics import ROOM_STEP_GRACE
from triforce.specialist import RewardConfig

from .test_env import make_env


def test_quiet_fight_penalty_respects_grace_and_stops_without_targetable_enemies(session):
    env = make_env(session, 'goriya', '1_44w_0', no_progress_steps=1000, max_steps=1000,
                   reward=RewardConfig(pbrs=False, quiet_enemy_penalty=0.05))
    env.reset(seed=0)
    for _ in range(100):
        if session.state.enemies and all(enemy.is_active for enemy in session.state.enemies):
            break
        env.step(Action.parse('WAIT 4').to_indices())
    assert session.state.enemies and all(enemy.is_active for enemy in session.state.enemies)
    session.emu.write(0x66C, 1)  # freeze the room's monsters, without inventing hits or progress
    session.apply_ram({})
    env.restore(session.snapshot(), '1_44w_0')

    for _ in range(ROOM_STEP_GRACE + 50):
        *_, terminated, truncated, info = env.step(Action.parse('WAIT 8').to_indices())
        assert not (terminated or truncated)
        assert 'penalty-quiet-enemies' not in info['rewards']
        if env.linger.tail_steps == ROOM_STEP_GRACE:
            break
    assert env.linger.tail_steps == ROOM_STEP_GRACE
    *_, info = env.step(Action.parse('WAIT 8').to_indices())
    assert env.linger.tail_steps == ROOM_STEP_GRACE + 1
    assert info['rewards']['penalty-quiet-enemies'] == -0.05

    for enemy in session.state.enemies:
        session.emu.write(0x34F + enemy.index, 0)
    session.emu.write(0x34C, 0)
    session.apply_ram({})
    *_, info = env.step(Action.parse('WAIT 8').to_indices())
    assert env.linger.tail_steps > ROOM_STEP_GRACE
    assert 'penalty-quiet-enemies' not in info['rewards']
    env.close()
