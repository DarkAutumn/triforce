"""ZeldaEnv end to end on real savestates: episodes, endings, outcomes, terminal rewards, info contract."""

import random

import numpy as np
import pytest

from triforce.actions import DIRECTIONS, DURATIONS, KINDS, Action, ActionMask
from triforce.end_conditions import UnclassifiedEndingError
from triforce.env import InvalidActionError, ZeldaEnv
from triforce.objectives import ObjectiveKind
from triforce.rewards import TERMINAL_FAILURE_PENALTY_VALUE, success_reward_value
from triforce.specialist import RamRandomization, load_specialist
from triforce.zelda_enums import ActionKind, Direction, MapLocation, SwordKind

STEP_INFO_KEYS = {'frames', 'interrupt', 'action', 'rewards', 'room', 'health', 'max_health', 'steps',
                  'total_frames', 'specialist'}


def make_env(session, specialist, start_state=None, **update):
    """An env on the shared session.  `update` overrides config fields; ram_randomization defaults to none so
    tests are deterministic."""
    config = load_specialist(specialist)
    update.setdefault('ram_randomization', RamRandomization())
    config = config.model_copy(update=update)
    return ZeldaEnv(config, session=session, start_state=start_state, capture_dir=None)


def random_legal_action(mask_flat, rng):
    """Uniform over legal kinds, then legal direction/duration for that kind."""
    mask = ActionMask.from_flat(mask_flat)
    kind = rng.choice([i for i in range(len(KINDS)) if mask.kind[i]])
    dirs = [i for i in range(len(DIRECTIONS)) if mask.direction[kind, i]] or [0]
    durs = [i for i in range(len(DURATIONS)) if mask.duration[kind, i]] or [0]
    return np.array([kind, rng.choice(dirs), rng.choice(durs)])


def greedy_step(env):
    """One env step of a 1-ply lookahead walker: tries every legal MOVE 8 on a session snapshot and takes the one
    that leaves the room through a wanted exit or gets closest on the PBRS wavefront (objective exits/targets)."""
    session = env.session
    wavefront = session.state.pbrs_wavefront
    wanted = session.state.objectives.next_rooms
    snap, orig, orig_mask = session.snapshot(), session.state, session.mask
    best = None
    for direction in session.mask.directions(ActionKind.MOVE):
        action = Action(ActionKind.MOVE, direction, 8)
        change = session.act(action)
        if change.gained_triforce:
            score = -1
        elif change.changed_location:
            score = -1 if change.state.full_location in wanted else 9999
        else:
            score = wavefront.get(change.state.link.tile, 999)
        session.restore(snap)
        if best is None or score < best[0]:
            best = (score, action)
    session.state, session.mask = orig, orig_mask
    orig.reactivate()
    return env.step(best[1].to_indices())


def run_until_done(env, policy, max_steps=400):
    """Steps until the episode ends; returns the list of (reward, info)."""
    results = []
    for _ in range(max_steps):
        _, reward, terminated, truncated, info = policy(env)
        results.append((reward, info))
        if terminated or truncated:
            return results
    raise AssertionError(f"episode did not end in {max_steps} steps (room {env.session.state.full_location})")


def test_spaces_and_reset_contract(session):
    env = make_env(session, 'd1-melee', '1_73n')
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert env.action_space.nvec.tolist() == [7, 8, 4]
    assert STEP_INFO_KEYS <= set(info) and info['specialist'] == 'd1-melee' and info['room'] == (1, 0x73, False)
    assert session.state.objectives.next_rooms == {MapLocation(1, 0x72, False), MapLocation(1, 0x74, False)}


def test_random_legal_episode_ends_classified(session):
    """d1-melee from a start state with random legal actions: every episode ends with a classified outcome,
    every step returns valid observations and the info contract."""
    env = make_env(session, 'd1-melee', '1_73n', max_steps=150)
    rng = random.Random(3)
    obs, _ = env.reset(seed=3)
    outcomes = []
    for _ in range(600):
        obs, reward, terminated, truncated, info = env.step(random_legal_action(obs['action_mask'], rng))
        assert STEP_INFO_KEYS <= set(info)
        assert np.isfinite(reward) and info['frames'] > 0
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            assert info['outcome'] in ('success', 'hard-failure', 'soft-failure')
            assert truncated == (info['ending'] == 'max-steps')
            outcomes.append(info['outcome'])
            if len(outcomes) == 2:
                break
            obs, _ = env.reset()
    assert outcomes, "no episode ended"


def test_correct_exit_is_success_with_health_scaled_reward(session):
    env = make_env(session, 'ow-start', '0_77n', ram_randomization=RamRandomization(hp=(0.5, 0.5)))
    env.reset(seed=0)
    assert session.state.link.health == 1.5
    reward, info = run_until_done(env, greedy_step)[-1]
    assert (info['ending'], info['outcome']) == ('exit-correct', 'success')
    assert info['rewards']['reward-terminal-success'] == pytest.approx(success_reward_value(1.5, 3)) == 15.0
    assert reward == pytest.approx(min(20.0, sum(info['rewards'].values())))


def test_full_health_success_pays_full_terminal(session):
    env = make_env(session, 'ow-start', '0_77n')
    env.reset(seed=0)
    _, info = run_until_done(env, greedy_step)[-1]
    assert info['rewards']['reward-terminal-success'] == pytest.approx(20.0)


def test_wrong_exit_is_soft_failure(session):
    env = make_env(session, 'ow-start', '0_77w')
    env.reset(seed=0)
    assert MapLocation(0, 0x76, False) not in session.state.objectives.next_rooms
    results = run_until_done(env, lambda e: e.step(Action.parse('MOVE W 16').to_indices()), max_steps=20)
    reward, info = results[-1]
    assert info['room'] == (0, 0x76, False)
    assert (info['ending'], info['outcome']) == ('exit-wrong', 'soft-failure')
    assert info['rewards']['penalty-terminal-failure'] == -TERMINAL_FAILURE_PENALTY_VALUE
    assert reward <= -19.0


def test_death_is_hard_failure(session):
    env = make_env(session, 'd1-wallmaster', '1_44w', ram_randomization=RamRandomization(hp=(0.01, 0.01)))   # rounds up to half a heart
    obs, _ = env.reset(seed=2)
    assert session.state.link.health == 0.5
    rng = random.Random(2)
    reward, info = run_until_done(env, lambda e: e.step(random_legal_action(e.session.mask.to_flat(), rng)))[-1]
    assert (info['ending'], info['outcome']) == ('death', 'hard-failure')
    assert reward == pytest.approx(-20.0 + sum(v for k, v in info['rewards'].items()
                                               if k != 'penalty-terminal-failure'), abs=1.0)
    assert all(v <= 0 for v in info['rewards'].values()), "hard failures drop positive rewards"


def test_unlisted_ending_raises(session):
    """Deny-by-default: an ow-start variant that only lists triforce-collected can't classify exit-correct."""
    env = make_env(session, 'ow-start', '0_77n', success=['triforce-collected'])
    env.reset(seed=0)
    with pytest.raises(UnclassifiedEndingError, match='exit-correct'):
        run_until_done(env, greedy_step)


def test_max_steps_truncates_without_terminal_reward(session):
    env = make_env(session, 'd1-melee', '1_73n', max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(Action.parse('WAIT 8').to_indices())
    assert truncated and not terminated
    assert (info['ending'], info['outcome']) == ('max-steps', 'soft-failure')
    assert not any(k.startswith(('reward-terminal', 'penalty-terminal')) for k in info['rewards'])
    with pytest.raises(RuntimeError):
        env.step(Action.parse('WAIT 8').to_indices())


def test_masked_action_raises(session):
    env = make_env(session, 'd1-melee', '1_73n')
    env.reset(seed=0)
    assert not session.mask.kind[ActionKind.SWORD]          # empty room: no weapons
    with pytest.raises(InvalidActionError):
        env.step(Action(ActionKind.SWORD, Direction.N).to_indices())


def test_close_keeps_borrowed_session(session):
    env = make_env(session, 'd1-melee', '1_73n')
    env.reset(seed=0)
    env.close()
    session.load('1_73n')
    assert session.state.location == 0x73


def test_start_state_outside_rooms_is_rejected(session):
    from triforce.start_states import StartStateError   # pylint: disable=import-outside-toplevel
    env = make_env(session, 'd1-melee', '1_44w')
    with pytest.raises(StartStateError, match='0x44'):
        env.reset(seed=0)


def test_sword_cave_objective_and_exit(session):
    """In the sword cave (no sword) the objective is the treasure; after taking it, S out of the cave is a correct
    exit that stays inside ow-start, and N from the start screen ends the episode with success."""
    env = make_env(session, 'ow-start', '0_77cave')
    env.reset(seed=0)
    objectives = session.state.objectives
    assert objectives.kind == ObjectiveKind.TREASURE and not objectives.next_rooms
    gained = None
    for _ in range(40):
        _, _, terminated, _, info = greedy_step(env)
        assert not terminated
        if session.state.link.sword != SwordKind.NONE:
            gained = info
            break
    assert gained is not None and 'reward-gained-sword' in gained['rewards']
    assert session.state.objectives.next_rooms == {MapLocation(0, 0x77, False)}

    for _ in range(60):
        _, _, terminated, _, info = env.step(Action.parse('MOVE S 16').to_indices())
        if info['room'] == (0, 0x77, False):
            break
    assert not terminated and 'reward-new-location' in info['rewards']
    _, info = run_until_done(env, greedy_step)[-1]
    assert (info['ending'], info['outcome']) == ('exit-correct', 'success')


def test_triforce_pickup_is_success_for_d1_melee(session):
    env = make_env(session, 'd1-melee', '1_36w')
    env.reset(seed=0)
    assert session.state.objectives.kind == ObjectiveKind.TREASURE
    _, info = run_until_done(env, greedy_step, max_steps=100)[-1]
    assert (info['ending'], info['outcome']) == ('triforce-collected', 'success')
