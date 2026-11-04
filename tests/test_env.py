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
from triforce.zelda_enums import ActionKind, Direction, MapLocation, SwordKind, TileIndex

STEP_INFO_KEYS = {'frames', 'interrupt', 'action', 'rewards', 'room', 'health', 'max_health', 'steps',
                  'total_frames', 'specialist', 'route'}


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


def chase_action(state):
    """MOVE 8 toward the nearest active enemy along the larger axis (to get hit)."""
    enemy = min(state.active_enemies, key=lambda e: e.distance, default=None)
    if enemy is None:
        return 'WAIT 8'
    dx, dy = enemy.position.x - state.link.position.x, enemy.position.y - state.link.position.y
    if abs(dx) > abs(dy):
        return 'MOVE E 8' if dx > 0 else 'MOVE W 8'
    return 'MOVE S 8' if dy > 0 else 'MOVE N 8'


def chase_step(env):
    """One env step toward the nearest enemy (WAIT when that move is masked)."""
    action = Action.parse(chase_action(env.session.state))
    if not env.session.mask.is_allowed(action):
        action = Action.parse('WAIT 8')
    return env.step(action.to_indices())


def test_spaces_and_reset_contract(session):
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0')
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert env.action_space.nvec.tolist() == [7, 8, 4]
    assert STEP_INFO_KEYS <= set(info) and info['specialist'] == 'stalfos-keese-gel'
    assert info['room'] == (1, 0x73, False)


def test_random_legal_episode_ends_classified(session):
    """stalfos-keese-gel from a start state with random legal actions: every episode ends with a classified outcome,
    every step returns valid observations and the info contract."""
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0', max_steps=150)
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
            hearts = info['hearts']
            assert hearts['dropped'] == hearts['taken'] + hearts['left'] + hearts['expired']
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
    env = make_env(session, 'goriya', '1_44w', ram_randomization=RamRandomization(hp=(0.01, 0.01)))   # rounds up to half a heart
    obs, _ = env.reset(seed=2)
    assert session.state.link.health == 0.5
    env.step(Action.parse('MOVE E 16').to_indices())
    reward, info = run_until_done(env, chase_step)[-1]
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
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0', max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(Action.parse('WAIT 8').to_indices())
    assert truncated and not terminated
    assert (info['ending'], info['outcome']) == ('max-steps', 'soft-failure')
    assert not any(k.startswith(('reward-terminal', 'penalty-terminal')) for k in info['rewards'])
    with pytest.raises(RuntimeError):
        env.step(Action.parse('WAIT 8').to_indices())



def test_no_room_progress_truncates_at_threshold_and_resets_shared_tracker(session):
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0', no_progress_steps=3, max_steps=100)
    env.reset(seed=0)
    tracker = env.linger
    for step in range(1, 4):
        _, _, terminated, truncated, info = env.step(Action.parse('WAIT 8').to_indices())
        assert not terminated
        assert truncated == (step == 3)
        assert tracker.tail_steps == step
    assert (info['ending'], info['outcome']) == ('no-room-progress', 'soft-failure')
    assert info['linger']['tail_steps'] == info['linger']['steps'] == 3
    assert not any(k.startswith(('reward-terminal', 'penalty-terminal')) for k in info['rewards'])
    with pytest.raises(RuntimeError):
        env.step(Action.parse('WAIT 8').to_indices())
    env.reset(seed=0)
    assert env.linger is tracker and tracker.tail_steps == 0
    *_, terminated, truncated, _ = env.step(Action.parse('WAIT 8').to_indices())
    assert not (terminated or truncated)


def test_no_room_progress_disabled_preserves_position_timeout(session):
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0', max_steps=100)
    env.reset(seed=0)
    for step in range(1, 51):
        *_, terminated, truncated, info = env.step(Action.parse('WAIT 8').to_indices())
        assert not truncated
        assert terminated == (step == 50)
    assert (info['ending'], info['outcome']) == ('stuck', 'hard-failure')


def test_no_room_progress_keeps_episode_cap(session):
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0', no_progress_steps=10, max_steps=2)
    env.reset(seed=0)
    env.step(Action.parse('WAIT 8').to_indices())
    *_, terminated, truncated, info = env.step(Action.parse('WAIT 8').to_indices())
    assert truncated and not terminated and info['ending'] == 'max-steps'


@pytest.mark.parametrize('progress', ['hit', 'pickup', 'heart', 'healing', 'approach', 'exit'])
def test_measured_ending_credits_threshold_action_before_check(progress):
    from types import SimpleNamespace as NS
    from triforce.end_conditions import NoRoomProgress, build_end_conditions, Timeout, classify_ending
    from triforce.linger import LingerTracker

    def state(tile=5, health=2, room=0x73):
        return NS(level=1, location=room, in_cave=False, full_location=(1, room, False), game_over=False,
                  objectives=NS(kind=ObjectiveKind.MOVE, next_rooms={0x63}, pbrs_targets={0}),
                  pbrs_wavefront={t: t for t in range(10)}, room_kills=0, enemies=[],
                  link=NS(tile=tile, keys=0, health=health, max_health=3))

    tracker = LingerTracker()
    prev = state()
    tracker.reset(prev)
    conditions = build_end_conditions([], [], linger=tracker, no_progress_steps=3)
    condition = conditions[-1]
    assert isinstance(condition, NoRoomProgress)
    assert not any(isinstance(c, Timeout) for c in conditions)
    assert isinstance(build_end_conditions([], [])[-1], Timeout)
    with pytest.raises(ValueError, match='shared LingerTracker'):
        build_end_conditions([], [], no_progress_steps=3)
    with pytest.raises(ValueError, match='positive'):
        build_end_conditions([], [], linger=tracker, no_progress_steps=0)

    def update(curr, **fields):
        change = NS(previous=prev, state=curr, frames=8, enemies_hit={}, items_gained=[],
                    gained_triforce=False, health_gained=0)
        change.__dict__.update(fields)
        tracker.update(change, hearts_taken=int(progress == 'heart' and curr is not prev))
        return condition.check(change)

    assert update(prev) is None
    assert update(prev) is None and tracker.tail_steps == 2
    curr = state(tile=4 if progress == 'approach' else 5, health=3 if progress == 'healing' else 2,
                 room=0x63 if progress == 'exit' else 0x73)
    assert update(curr, enemies_hit={1: 1} if progress == 'hit' else {},
                  items_gained=[1] if progress == 'pickup' else [],
                  health_gained=1 if progress == 'healing' else 0) is None
    assert tracker.tail_steps == 0
    prev = curr
    assert update(prev) is None
    assert update(prev) is None
    ending = update(prev)
    assert ending == 'no-room-progress' and classify_ending(ending, []) == 'soft-failure'
    condition.clear()
    tracker.reset(curr)
    assert update(prev) is None and tracker.tail_steps == 1

def test_masked_action_raises(session):
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0')
    env.reset(seed=0)
    env.step(Action.parse('MOVE N 16').to_indices())        # off the S door tile, where weapons are masked anyway
    assert not session.mask.kind[ActionKind.SWORD]          # empty room: no weapons
    with pytest.raises(InvalidActionError):
        env.step(Action(ActionKind.SWORD, Direction.N).to_indices())


def test_close_keeps_borrowed_session(session):
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0')
    env.reset(seed=0)
    env.close()
    session.load('1_73s_0')
    assert session.state.location == 0x73


def test_start_state_outside_rooms_is_rejected(session):
    from triforce.start_states import StartStateError   # pylint: disable=import-outside-toplevel
    env = make_env(session, 'stalfos-keese-gel', '1_44w')
    with pytest.raises(StartStateError, match='0x44'):
        env.reset(seed=0)


def test_sword_cave_objective_and_exit(session):
    """In the sword cave (no sword) the objective is the treasure; after taking it, S out of the cave is a correct
    exit that stays inside ow-start, and N from the start screen ends the episode with success."""
    env = make_env(session, 'ow-start', '0_77cave_0')
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


def test_triforce_pickup_is_success_in_the_triforce_room(session):
    env = make_env(session, 'stalfos-keese-gel', '1_36w')
    env.reset(seed=0)
    assert session.state.objectives.kind == ObjectiveKind.TREASURE
    _, info = run_until_done(env, greedy_step, max_steps=100)[-1]
    assert (info['ending'], info['outcome']) == ('triforce-collected', 'success')


@pytest.mark.parametrize('bombs_on', [True, False])
def test_items_config_masks_bombs_even_when_link_has_them(session, bombs_on):
    env = make_env(session, 'goriya', '1_44w', ram_randomization=RamRandomization(bombs=(4, 4)),
                   items=load_specialist('goriya').items.model_copy(update={'bombs': bombs_on}))
    env.reset(seed=0)
    for _ in range(10):         # off the door tile (items can't be used there) and the Goriyas out of their clouds
        obs = env.step(Action.parse('MOVE E 16').to_indices())[0]
        if session.state.link.position.x >= 32 and session.state.active_enemies:
            break
    assert session.state.link.bombs == 4 and session.state.active_enemies
    assert bool(session.mask.kind[ActionKind.BOMB]) == bombs_on
    assert bool(ActionMask.from_flat(obs['action_mask']).kind[ActionKind.BOMB]) == bombs_on


def test_route_shutter_requires_combat_without_a_room_treasure(session):
    env = make_env(session, 'd2-rope', '2_2es_0')
    env.reset(seed=0)
    objective = session.state.objectives
    assert objective.kind == ObjectiveKind.FIGHT
    assert not objective.next_rooms


@pytest.mark.parametrize(('start_state', 'next_room'), [('2_6es_0', 0x5e), ('2_6ds_0', 0x6e)])
def test_open_route_ignores_optional_combat(session, start_state, next_room):
    env = make_env(session, 'd2-rope', start_state)
    env.reset(seed=0)
    objective = session.state.objectives
    assert objective.kind == ObjectiveKind.MOVE
    assert objective.next_rooms == {MapLocation(2, next_room, False)}


@pytest.mark.parametrize('keys', [0, 1])
def test_route_room_only_offers_its_route_exit(session, keys):
    """0x43's route exit is the locked E door to 0x44.  Without a key the shortest path to the triforce goes N for
    the Stalfos room's key (0x33, off the route); the objective still only accepts 0x44."""
    env = make_env(session, 'stalfos-keese-gel', '1_43w_0', ram_randomization=RamRandomization(keys=(keys, keys)))
    env.reset(seed=0)
    assert session.state.link.keys == keys
    assert session.state.objectives.next_rooms == {MapLocation(1, 0x44, False)}


def test_first_visit_to_0x73_follows_the_route_order(session):
    """Entering Dungeon 1 with no keys, 0x72 (W) and 0x74 (E) are equally short key detours; the route takes 0x72
    first, and a policy that went E first met a 0x73-from-E-needing-W case it never trained on (r4 chain)."""
    env = make_env(session, 'stalfos-keese-gel', '1_73s_0')
    env.reset(seed=0)
    assert session.state.link.keys == 0
    assert session.state.objectives.next_rooms == {MapLocation(1, 0x72, False)}


def test_enter_room_success_needs_the_objective(session):
    """ow-octorok lists enter-room:1:0x73 (the D1 stairs in 0x37) as success.  After Dungeon 1 (0_37c_2: Link
    just came out of those stairs with the triforce) the objective wants E, so walking back down is exit-wrong."""
    env = make_env(session, 'ow-octorok', '0_37c_2')
    env.reset(seed=0)
    assert session.state.objectives.next_rooms == {MapLocation(0, 0x38, False)}
    for action in ['MOVE S 16', 'MOVE S 16', 'MOVE N 16', 'MOVE N 16']:
        *_, terminated, truncated, info = env.step(Action.parse(action).to_indices())
        if terminated or truncated:
            break
    assert info['room'] == (1, 0x73, False)
    assert (info['ending'], info['outcome']) == ('exit-wrong', 'soft-failure')


@pytest.mark.parametrize('route, wanted', [('milestone1', 0x28), ('milestone1-fairy', 0x48)])
def test_after_dungeon_1_each_route_leaves_0x38_its_own_way(session, route, wanted):
    """0x38 from W, just out of Dungeon 1: milestone1 goes N toward D2, milestone1-fairy S toward the fairy pond."""
    env = make_env(session, 'ow-zora', '0_38w_2', route=route)
    _, info = env.reset(seed=0)
    assert info['route'] == route
    assert session.state.objectives.next_rooms == {MapLocation(0, wanted, False)}


def test_a_room_the_routes_leave_differently_trains_both(session):
    """Without a pinned route, 0x38-from-W resets play either route; rooms where the routes agree play milestone1."""
    env = make_env(session, 'ow-zora', '0_38w_2')
    seen = set()
    for seed in range(16):
        _, info = env.reset(seed=seed)
        seen.add((info['route'], frozenset(session.state.objectives.next_rooms)))
    assert seen == {('milestone1', frozenset({MapLocation(0, 0x28, False)})),
                    ('milestone1-fairy', frozenset({MapLocation(0, 0x48, False)}))}
    _, info = make_env(session, 'ow-octorok', '0_37c_2').reset(seed=0)
    assert info['route'] == 'milestone1'


def test_fairy_pond_holds_the_exit_until_the_hearts_are_full(session):
    """0x39 (milestone1-fairy's waypoint): below full hearts no exit is correct and the pond edge is the target;
    walking N to it fills every heart (Z_04 UpdatePondFairy), and then S is the exit."""
    env = make_env(session, 'ow-start', '0_39s_2', ram_randomization=RamRandomization(hp=(0.25, 0.25)))
    _, info = env.reset(seed=0)
    link = session.state.link
    assert info['route'] == 'milestone1-fairy' and link.health < link.max_health
    assert session.state.objectives.kind == ObjectiveKind.TREASURE and not session.state.objectives.next_rooms
    for _ in range(12):
        *_, terminated, truncated, info = env.step(Action.parse('MOVE N 8').to_indices())
        assert not (terminated or truncated)
        if session.state.link.health == session.state.link.max_health:
            break
    assert session.state.link.health == session.state.link.max_health
    assert session.state.objectives.next_rooms == {MapLocation(0, 0x49, False)}


@pytest.mark.parametrize('state, wanted', [('0_49s_2', 0x39), ('0_49n_2', 0x59)])
def test_fairy_route_heads_to_the_pond_then_east(session, state, wanted):
    """0x49 from S goes N to the pond; back from the pond (from N) the waypoint is done and the route goes S/E."""
    env = make_env(session, 'ow-octorok', state)
    _, info = env.reset(seed=0)
    assert info['route'] == 'milestone1-fairy'
    assert session.state.objectives.next_rooms == {MapLocation(0, wanted, False)}


def test_wavefront_has_no_lane_below_the_bottom_tree_line(session):
    """0x5c's lower-left pocket only opens S.  Link is 2 tiles tall, so the wavefront from the E exit must not reach
    it along the off-map row under the bottom trees (before round 6 it did, and PBRS pulled Link into the trees)."""
    state = session.load('0_5cw_0', fresh_room=True)
    wavefront = state.room.calculate_wavefront_for_link(state.room.exits[Direction.E])
    assert wavefront.get(state.link.tile) is not None        # the upper corridor reaches the E exit
    assert wavefront.get(TileIndex(2, 16)) is None            # the pocket doesn't


@pytest.mark.parametrize('one_room', [True, False])
def test_one_room_episode_ends_at_the_first_correct_exit(session, one_room):
    """1_73e (2 keys) exits N through the locked door into 0x63, another stalfos-keese-gel room: one_room ends the
    episode there as exit-correct, otherwise it continues."""
    env = make_env(session, 'stalfos-keese-gel', '1_73e', one_room=one_room)
    env.reset(seed=0)
    results = []
    while not any(info['room'] == (1, 0x63, False) for _, info in results):
        assert len(results) < 60 and not (results and 'ending' in results[-1][1])
        _, reward, *_, info = greedy_step(env)
        results.append((reward, info))
    info = results[-1][1]
    assert info.get('ending') == ('exit-correct' if one_room else None)
