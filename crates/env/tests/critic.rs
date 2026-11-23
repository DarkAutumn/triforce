//! Each critic term and end condition on synthetic steps (hand-built RAM images, no emulator except the
//! puzzle-door case): the term fires on its condition, stays silent otherwise, and is clamped/decayed as
//! documented; end conditions fire in priority order.

mod common;

use env::SpecialistConfig;
use env::critic::{
    BossTransitions, COMBAT_DECAY_THRESHOLD, Critique, GameplayCritic, ROOM_STEP_GRACE, ROOM_STEP_PENALTY_MAX,
    ROOM_STEP_PENALTY_MIN, ROOM_STEP_RAMP,
};
use env::end_conditions::{
    BOSS_DEAD_AND_HEART, DEATH, EXIT_CORRECT, EXIT_WRONG, EndConditions, MAX_STEPS, NO_PROGRESS,
    NO_DAMAGE, NO_ROOM_PROGRESS, STUCK, StepView, TRIFORCE_COLLECTED, WALLMASTERED, classify_ending,
    enter_room_name,
};
use env::heart_value::HeartContext;
use env::rewards::{
    HARD_FAILURE, REWARD_LARGE, REWARD_MAXIMUM, REWARD_MEDIUM, REWARD_MINIMUM, REWARD_SMALL,
    REWARD_TINY, StepRewards, score_step,
};
use indexmap::IndexMap;
use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};
use zelda::enums::enemy;
use zelda::objectives::{Objective, ObjectiveKind};
use zelda::room::{Target, Wavefront};
use zelda::tables::{self, RAM_SIZE};
use zelda::{Action, ActionMask, Change, GameMap, GameView, MapLocation, Observed, Var};

static MAP: LazyLock<Arc<GameMap>> = LazyLock::new(|| Arc::new(GameMap::builtin()));

const ROOM: MapLocation = MapLocation::new(1, 0x44, false);
const EAST: MapLocation = MapLocation::new(1, 0x45, false);

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// A playable dungeon RAM image: room 0x44, Link at (120, 141), 3 of 3 hearts, wood sword.
#[derive(Clone)]
struct Ram([u8; RAM_SIZE]);

impl Ram {
    fn new() -> Ram {
        let mut r = Ram([0; RAM_SIZE]);
        r.set(Var::Level, ROOM.level)
            .set(Var::Location, ROOM.value)
            .set(Var::Mode, 5)
            .set(Var::LinkX, 120)
            .set(Var::LinkY, 141)
            .set(Var::HeartsAndContainers, 0x22)
            .set(Var::PartialHearts, 0xFF)
            .set(Var::Sword, 1)
            .set(Var::TreasureFlag, 0xFF);
        r
    }

    fn set(&mut self, var: Var, value: u8) -> &mut Ram {
        self.0[var.addr() as usize] = value;
        self
    }

    fn with(&self, var: Var, value: u8) -> Ram {
        let mut r = self.clone();
        r.set(var, value);
        r
    }

    fn room(&self, location: MapLocation) -> Ram {
        self.with(Var::Level, location.level)
            .with(Var::Location, location.value)
    }

    /// Enemy `id` in `slot` at (x, y) with one heart.
    fn enemy(&mut self, slot: usize, id: u8, x: u8, y: u8) -> &mut Ram {
        self.0[tables::OBJ_ID.0 + slot] = id;
        self.0[tables::OBJ_POS_X.0 + slot] = x;
        self.0[tables::OBJ_POS_Y.0 + slot] = y;
        self.0[tables::OBJ_HEALTH.0 + slot] = 0x10;
        self
    }

    fn link(&self, x: u8, y: u8) -> Ram {
        self.with(Var::LinkX, x).with(Var::LinkY, y)
    }

    fn view(&self) -> Arc<GameView> {
        Arc::new(GameView::new(&self.0, 0, false, MAP.clone()))
    }
}

/// One step from `prev` to `curr` (health deltas as the session computes them for equal max health).
fn step(prev: &Ram, curr: &Ram, action: &str) -> Change {
    let (previous, state) = (prev.view(), curr.view());
    let delta = state.health() - previous.health();
    Change {
        action: Action::parse(action).unwrap(),
        frames: 8,
        menu_frames: 0,
        interrupt: None,
        control_frames: 8,
        projectile_blocks: Vec::new(),
        clip: None,
        clip_finished_by_move: false,
        previous,
        state,
        action_mask: ActionMask::default(),
        health_lost: (-delta).max(0.0),
        health_gained: delta.max(0.0),
        observed: Observed::default(),
        enemies_hit: Vec::new(),
        enemies_stunned: Vec::new(),
        items_gained: Vec::new(),
    }
}

/// A step whose `hits` landed (observed) and are credited.
fn hits(mut change: Change, hits: &[(u8, i32)]) -> Change {
    change.enemies_hit = hits.to_vec();
    change.observed.enemies_hit = hits.to_vec();
    change
}

/// A step credited with `hits` a look-ahead predicted; nothing landed yet.
fn predicted(mut change: Change, hits: &[(u8, i32)]) -> Change {
    change.enemies_hit = hits.to_vec();
    change
}

fn objective(next_rooms: &[MapLocation]) -> Objective {
    Objective {
        kind: ObjectiveKind::Move,
        targets: BTreeSet::new(),
        next_rooms: next_rooms.iter().copied().collect(),
        pbrs_targets: BTreeSet::new(),
        retreats: BTreeSet::new(),
        clips: false,
        treasure_refused: false,
        drops_first: false,
        bag_drops: false,
        labels: Vec::new(),
    }
}

/// The critic's rewards for one step (`objective` for both states, no wavefront, no quiet steps).
fn critique_with(
    critic: &mut GameplayCritic,
    change: &Change,
    objective: &Objective,
    wavefront: Option<&Wavefront>,
    quiet_steps: u32,
) -> StepRewards {
    let c = Critique {
        change,
        map: &MAP,
        prev_objective: objective,
        objective,
        prev_wavefront: wavefront,
        quiet_steps,
    };
    let mut rewards = StepRewards::default();
    critic.critique(&c, &mut rewards);
    rewards
}

fn critique(critic: &mut GameplayCritic, change: &Change) -> StepRewards {
    critique_with(critic, change, &objective(&[EAST]), None, 0)
}

fn critic() -> GameplayCritic {
    GameplayCritic::new(false, 1.0, 0.0)
}

fn names(rewards: &StepRewards) -> Vec<&'static str> {
    rewards.iter().map(|o| o.name).collect()
}

fn value(rewards: &StepRewards, name: &str) -> f64 {
    rewards
        .get(name)
        .unwrap_or_else(|| panic!("{name} missing: {:?}", names(rewards)))
        .value
}

// --- attacks ---

#[test]
fn hits_misses_and_cave_attacks() {
    let mut base = Ram::new();
    base.enemy(3, enemy::STALFOS, 40, 100);
    let r = critique(
        &mut critic(),
        &hits(step(&base, &base, "SWORD N"), &[(3, 1)]),
    );
    assert_eq!(names(&r), ["reward-hit"]);
    assert_eq!(value(&r, "reward-hit"), REWARD_SMALL);

    let r = critique(&mut critic(), &step(&base, &base, "SWORD N"));
    assert_eq!(names(&r), ["penalty-attack-miss"]);
    assert_eq!(value(&r, "penalty-attack-miss"), -REWARD_MINIMUM);
    for action in ["MOVE N 8", "WAIT 8", "BOMB N", "CLIP N"] {
        let r = critique(&mut critic(), &step(&base, &base, action));
        assert!(!r.contains("penalty-attack-miss"), "{action}");
    }

    // Hitting anything in a cave (the old man, a shopkeeper) is punished, not rewarded.
    let cave = base.with(Var::Mode, 11);
    let r = critique(
        &mut critic(),
        &hits(step(&cave, &cave, "SWORD N"), &[(3, 1)]),
    );
    assert_eq!(names(&r), ["penalty-attack-cave"]);
    assert_eq!(value(&r, "penalty-attack-cave"), -REWARD_MAXIMUM);
}

#[test]
fn combat_rewards_decay_per_room_after_the_threshold() {
    let mut base = Ram::new();
    base.enemy(3, enemy::STALFOS, 40, 100);
    let mut critic = critic();
    let hit = |critic: &mut GameplayCritic, ram: &Ram| {
        value(
            &critique(critic, &hits(step(ram, ram, "SWORD N"), &[(3, 1)])),
            "reward-hit",
        )
    };
    // Full value while at most COMBAT_DECAY_THRESHOLD events came before in the room; then halving.
    for _ in 0..=COMBAT_DECAY_THRESHOLD {
        assert_eq!(hit(&mut critic, &base), REWARD_SMALL);
    }
    assert_eq!(hit(&mut critic, &base), REWARD_SMALL * 0.5);
    assert_eq!(hit(&mut critic, &base), REWARD_SMALL * 0.25);
    // Another room has its own count, and a new episode starts over.
    assert_eq!(hit(&mut critic, &base.room(EAST)), REWARD_SMALL);
    critic.clear();
    assert_eq!(hit(&mut critic, &base), REWARD_SMALL);
}

#[test]
fn boss_hits_scale_with_the_hit_count_and_replace_the_hit_reward() {
    let mut base = Ram::new();
    base.enemy(1, enemy::AQUAMENTUS, 160, 112)
        .enemy(3, enemy::STALFOS, 40, 100);
    let r = critique(
        &mut critic(),
        &hits(step(&base, &base, "SWORD E"), &[(1, 1)]),
    );
    assert_eq!(names(&r), ["reward-boss-hit"]);
    assert_eq!(value(&r, "reward-boss-hit"), REWARD_LARGE);
    // Two boss hits would be 1.5: clamped to the per-term maximum.
    let r = critique(
        &mut critic(),
        &hits(step(&base, &base, "SWORD E"), &[(1, 2)]),
    );
    assert_eq!(value(&r, "reward-boss-hit"), REWARD_MAXIMUM);
    // The boss killed and gone this step still counts (looked up in the previous state).
    let mut dead = base.clone();
    dead.0[tables::OBJ_ID.0 + 1] = 0;
    let r = critique(
        &mut critic(),
        &hits(step(&base, &dead, "SWORD E"), &[(1, 1)]),
    );
    assert_eq!(names(&r), ["reward-boss-hit"]);
}

/// Dodongo in slot 1 with ObjState `status`, stun timer `stun` and `hits` bombs swallowed.
fn dodongo(status: u8, stun: u8, hits: u8) -> Ram {
    let mut ram = Ram::new();
    ram.enemy(1, enemy::DODONGO2, 112, 141);
    ram.0[tables::OBJ_STATUS.0 + 1] = status;
    ram.0[tables::OBJ_STUN_TIMER.0 + 1] = stun;
    ram.0[tables::DODONGO_BOMB_HITS.0 + 1] = hits;
    ram
}

#[test]
fn dodongo_stun_pays_and_a_swallowed_bomb_costs() {
    let walking = dodongo(0, 0, 0);
    // The stun usually lands a step after the bomb (48-frame fuse): whatever the action, entering state 2 pays.
    let r = critique(&mut critic(), &step(&walking, &dodongo(2, 0, 0), "MOVE N 8"));
    assert_eq!(value(&r, "reward-boss-stunned"), REWARD_LARGE);
    // A swallow is the bomb-hit counter going up (the game raises it on the first bloated frame).
    let r = critique(&mut critic(), &step(&walking, &dodongo(1, 0, 1), "WAIT 8"));
    assert_eq!(value(&r, "penalty-boss-ate-bomb"), -REWARD_MEDIUM);
    // `reward.boss_ate_bomb_penalty` sets the magnitude; 0 drops the term.
    let mut strict = critic();
    strict.boss_ate_bomb_penalty = 1.0;
    let r = critique(&mut strict, &step(&walking, &dodongo(1, 0, 1), "WAIT 8"));
    assert_eq!(value(&r, "penalty-boss-ate-bomb"), -1.0);
    strict.boss_ate_bomb_penalty = 0.0;
    let r = critique(&mut strict, &step(&walking, &dodongo(1, 0, 1), "WAIT 8"));
    assert!(!names(&r).contains(&"penalty-boss-ate-bomb"));
    // Staying stunned (the timer set to $20 after the stun frame, then counting down), bloating on, or
    // recovering is neither.
    for (before, after) in [
        (dodongo(2, 0, 0), dodongo(2, 0x20, 0)),
        (dodongo(2, 0x20, 0), dodongo(2, 0x1C, 0)),
        (dodongo(1, 0, 1), dodongo(1, 0, 1)),
        (dodongo(2, 2, 0), walking.clone()),
    ] {
        let r = critique(&mut critic(), &step(&before, &after, "WAIT 8"));
        assert!(!names(&r).iter().any(|n| n.contains("boss")), "{:?}", names(&r));
    }
}

/// Digdogger (round 24): the recorder that starts the big one's split (FluteUsedState $51B rising from 0 with a big
/// Digdogger in the room) pays `reward-recorder-split` once per room; a recorder use is never an attack miss; the
/// small one's hits are boss hits.
#[test]
fn digdogger_recorder_split_pays_once_and_the_small_one_is_the_boss() {
    const FLUTE_USED_STATE: usize = 0x51B;
    let mut big = Ram::new();
    big.enemy(1, enemy::DIGDOGGER2, 112, 100);
    let mut heard = big.clone();
    heard.0[FLUTE_USED_STATE] = 1;
    let mut once = critic();
    let r = critique(&mut once, &step(&big, &heard, "RECORDER"));
    assert_eq!(names(&r), ["reward-recorder-split"]);
    assert_eq!(value(&r, "reward-recorder-split"), REWARD_LARGE);
    // Once per room and episode; a new episode pays again.
    let r = critique(&mut once, &step(&big, &heard, "RECORDER"));
    assert!(names(&r).is_empty(), "{:?}", names(&r));
    once.clear();
    assert_eq!(names(&critique(&mut once, &step(&big, &heard, "RECORDER"))), ["reward-recorder-split"]);
    // A tune that took no hold (the room had heard it), or none without a big Digdogger: nothing, and no miss.
    for (before, after) in [(heard.clone(), heard.clone()), (Ram::new(), Ram::new())] {
        let r = critique(&mut critic(), &step(&before, &after, "RECORDER"));
        assert!(names(&r).is_empty(), "{:?}", names(&r));
    }
    let mut small = Ram::new();
    small.enemy(1, enemy::DIGDOGGER_CHILD, 112, 100);
    let r = critique(&mut critic(), &hits(step(&small, &small, "SWORD E"), &[(1, 1)]));
    assert_eq!(names(&r), ["reward-boss-hit"]);
}

/// Events inside one step still count: bloated, back to walking and stunned (1 -> 2 at the boundaries), a stun
/// that ended and restarted (the timer goes up), and each of two Dodongos.
#[test]
fn boss_transitions_count_events_within_a_step() {
    let of = |before: &Ram, after: &Ram| BossTransitions::of(&step(before, after, "WAIT 16"));
    assert_eq!(of(&dodongo(1, 0, 1), &dodongo(2, 0, 1)), BossTransitions { stuns: 1, eats: 0 });
    assert_eq!(of(&dodongo(0, 0, 0), &dodongo(2, 0x20, 1)), BossTransitions { stuns: 1, eats: 1 });
    assert_eq!(of(&dodongo(2, 2, 0), &dodongo(2, 0x20, 0)), BossTransitions { stuns: 1, eats: 0 });
    let mut walking = Ram::new();
    walking.enemy(1, enemy::DODONGO, 112, 141).enemy(2, enemy::DODONGO2, 64, 141);
    let mut after = walking.clone();
    after.0[tables::OBJ_STATUS.0 + 1] = 2;
    after.0[tables::OBJ_STATUS.0 + 2] = 1;
    after.0[tables::DODONGO_BOMB_HITS.0 + 2] = 1;
    let change = step(&walking, &after, "WAIT 8");
    assert_eq!(BossTransitions::of(&change), BossTransitions { stuns: 1, eats: 1 });
    let r = critique(&mut critic(), &change);
    assert_eq!(value(&r, "reward-boss-stunned"), REWARD_LARGE);
    assert_eq!(value(&r, "penalty-boss-ate-bomb"), -REWARD_MEDIUM);
    assert_eq!(of(&after, &walking), BossTransitions::default());
}

#[test]
fn beam_hits_and_long_shots() {
    let mut base = Ram::new();
    base.enemy(3, enemy::STALFOS, 120, 60) // 81 pixels above Link
        .enemy(4, enemy::STALFOS, 120, 121); // 20 pixels above
    let flying = base.with(Var::BeamAnimation, 16);
    let far = critique(
        &mut critic(),
        &hits(step(&base, &flying, "BEAMS N"), &[(3, 1)]),
    );
    // The hit kills it (one heart), 81 pixels out: a kill at range (beyond the room's top, so no shot is lined up).
    assert_eq!(names(&far), ["reward-beam-hit", "reward-beam-distance", "reward-beam-kill"]);
    assert_eq!(value(&far, "reward-beam-hit"), REWARD_MEDIUM);
    assert_eq!(value(&far, "reward-beam-distance"), REWARD_TINY);
    let near = critique(
        &mut critic(),
        &hits(step(&base, &flying, "BEAMS N"), &[(4, 1)]),
    );
    assert_eq!(names(&near), ["reward-beam-hit"]);
    // Beams weren't available before the step (not full health): an ordinary hit.
    let hurt = base.with(Var::PartialHearts, 0x7F);
    let c = hits(
        step(&hurt, &hurt.with(Var::BeamAnimation, 16), "SWORD N"),
        &[(3, 1)],
    );
    assert_eq!(names(&critique(&mut critic(), &c)), ["reward-hit"]);
}

/// Round 38 (the owner: "more reward for long range kills (out of melee range), and possibly a small penalty for
/// ignoring a lined up shot [...] reward timing beams that WOULD intersect even if the enemy turns the other way"):
/// the projection carries each monster straight on (ObjDir, ObjQSpeedFrac), the shot made 12 frames after the press
/// 16 pixels ahead at 3 a frame.
#[test]
fn beams_at_range_kill_timing_and_unfired_shots() {
    const QSPEED: usize = 0x3BC;
    let flying = |r: &Ram| r.with(Var::BeamAnimation, 16);
    // A Stalfos 45 pixels north of Link (120, 141), standing.
    let mut north = Ram::new();
    north.enemy(3, enemy::STALFOS, 120, 96);
    assert_eq!(zelda::beams::lined_up(&north.view()), [zelda::Direction::N]);
    // Lined up, ready, and walking away: the small cost; a shot north: the timed shot (and its hit, and its kill).
    let r = critique(&mut critic(), &step(&north, &north.link(124, 141), "MOVE E 4"));
    assert_eq!(names(&r), ["penalty-beam-unfired"]);
    assert_eq!(value(&r, "penalty-beam-unfired"), -2.0 * REWARD_MINIMUM);
    let r = critique(&mut critic(), &hits(step(&north, &flying(&north), "BEAMS N"), &[(3, 1)]));
    assert_eq!(names(&r), ["reward-beam-hit", "reward-beam-kill", "reward-beam-timed"]);
    assert_eq!(value(&r, "reward-beam-kill"), REWARD_MEDIUM);
    assert_eq!(value(&r, "reward-beam-timed"), 2.0 * REWARD_TINY);
    // The shot meets it on its course but it turned away (no hit credited): the timed shot still pays.
    let r = critique(&mut critic(), &step(&north, &flying(&north), "BEAMS N"));
    assert_eq!(names(&r), ["penalty-attack-miss", "reward-beam-timed"]);
    // A shot the other way: neither the timing nor the cost.
    let r = critique(&mut critic(), &step(&north, &flying(&north), "BEAMS S"));
    assert_eq!(names(&r), ["penalty-attack-miss"]);
    // A hit that doesn't kill (two hearts): no kill reward.
    let mut tough = north.clone();
    tough.0[tables::OBJ_HEALTH.0 + 3] = 0x20;
    let r = critique(&mut critic(), &hits(step(&tough, &flying(&tough), "BEAMS N"), &[(3, 1)]));
    assert_eq!(names(&r), ["reward-beam-hit", "reward-beam-timed"]);
    // Not ready (hurt: no beams; a shot already flying): nothing lined up, nothing charged.
    let hurt = north.with(Var::PartialHearts, 0x7F);
    assert!(critique(&mut critic(), &step(&hurt, &hurt.link(124, 141), "MOVE E 4")).is_empty());
    let busy = flying(&north);
    assert!(zelda::beams::lined_up(&busy.view()).is_empty());
    // In melee range: not a shot at range.
    let mut close = Ram::new();
    close.enemy(3, enemy::STALFOS, 120, 120);
    assert!(zelda::beams::lined_up(&close.view()).is_empty());
    // Off the line now, walking onto it at 2 pixels a frame (ObjDir E, q-speed 128): lined up by its course; standing
    // there, not.
    let mut coming = Ram::new();
    coming.enemy(3, enemy::STALFOS, 80, 96);
    assert!(zelda::beams::lined_up(&coming.view()).is_empty());
    coming.0[tables::OBJ_DIRECTION.0 + 3] = 1;
    coming.0[QSPEED + 3] = 128;
    assert_eq!(zelda::beams::lined_up(&coming.view()), [zelda::Direction::N]);
    // A Darknut facing the shot takes no damage from it (asm:Damage_HandleWeaponHit@CheckDarknut); one facing aside
    // does.
    let mut darknut = Ram::new();
    darknut.enemy(3, enemy::RED_DARKNUT, 120, 96);
    darknut.0[tables::OBJ_DIRECTION.0 + 3] = 4;
    assert!(zelda::beams::lined_up(&darknut.view()).is_empty(), "facing south, at Link");
    darknut.0[tables::OBJ_DIRECTION.0 + 3] = 1;
    assert_eq!(zelda::beams::lined_up(&darknut.view()), [zelda::Direction::N]);
    // Immune to the sword (ObjInvincibilityMask bit 1): not lined up.
    let mut immune = north.clone();
    immune.0[0x4B2 + 3] = 0x01;
    assert!(zelda::beams::lined_up(&immune.view()).is_empty());
    // Each term off by its knob.
    let mut off = critic();
    off.beam_unfired_penalty = 0.0;
    off.beam_timed_reward = 0.0;
    off.beam_kill_reward = 0.0;
    assert!(critique(&mut off, &step(&north, &north.link(124, 141), "MOVE E 4")).is_empty());
    let r = critique(&mut off, &hits(step(&north, &flying(&north), "BEAMS N"), &[(3, 1)]));
    assert_eq!(names(&r), ["reward-beam-hit"]);
}

/// The visit's shot and kill counts (`zelda::beams::BeamStats`, chain room records and evaluation visits).
#[test]
fn beam_stats_count_shots_lined_decisions_and_kills_by_range() {
    use zelda::beams::BeamStats;
    let killed = |mut c: Change, slot: u8| {
        c.observed.enemies_killed = vec![slot];
        c
    };
    let mut north = Ram::new();
    north.enemy(3, enemy::STALFOS, 120, 96);
    let flying = north.with(Var::BeamAnimation, 16);
    let mut s = BeamStats::default();
    s.update(&step(&north, &north.link(124, 141), "MOVE E 4"));
    s.update(&step(&north, &flying, "BEAMS N"));
    s.update(&killed(step(&flying, &flying, "WAIT 4"), 3));
    assert_eq!((s.lined, s.lined_unfired, s.shots, s.shots_lined), (2, 1, 1, 1));
    assert_eq!((s.kills_far, s.kills_near, s.kills_other, s.kill_distance), (1, 0, 0, 45));
    // In melee range on a sword step: near; far without a shot flying (a bomb, an arrow): other.
    let mut close = Ram::new();
    close.enemy(3, enemy::STALFOS, 120, 121);
    s.update(&killed(step(&close, &close, "SWORD N"), 3));
    s.update(&killed(step(&north, &north, "BOMB N"), 3));
    assert_eq!((s.kills_far, s.kills_near, s.kills_other, s.kill_distance), (1, 1, 1, 65));
    // Hearts lost, and the beams with them.
    s.update(&step(&close, &close.with(Var::PartialHearts, 0x7F), "WAIT 4"));
    assert_eq!((s.hearts_lost, s.beams_lost), (0.5, 1));
}

#[test]
fn boomerang_stuns_pay_once_per_enemy_and_room() {
    let mut base = Ram::new();
    base.enemy(3, enemy::RED_GORIYA, 40, 100)
        .enemy(4, enemy::RED_GORIYA, 60, 100);
    let stun = |slots: &[u8], ram: &Ram| {
        let mut c = step(ram, ram, "BOOMERANG W");
        c.enemies_stunned = slots.to_vec();
        c
    };
    let mut critic = critic();
    let r = critique(&mut critic, &stun(&[3], &base));
    assert_eq!(names(&r), ["reward-boomerang-stun"]);
    assert_eq!(value(&r, "reward-boomerang-stun"), REWARD_SMALL);
    // The same Goriya again: nothing (and no miss penalty either, the throw did stun).
    assert!(critique(&mut critic, &stun(&[3], &base)).is_empty());
    assert_eq!(
        names(&critique(&mut critic, &stun(&[3, 4], &base))),
        ["reward-boomerang-stun"]
    );
    // Slot 3 in another room is another enemy.
    assert_eq!(
        names(&critique(&mut critic, &stun(&[3], &base.room(EAST)))),
        ["reward-boomerang-stun"]
    );
    // A stun from a sword swing is not a boomerang stun.
    let mut c = step(&base, &base, "SWORD W");
    c.enemies_stunned = vec![4];
    assert!(critique(&mut GameplayCritic::new(false, 1.0, 0.0), &c).is_empty());
}

#[test]
fn placing_a_bomb_costs_and_its_hits_pay() {
    let base = Ram::new().with(Var::Bombs, 3);
    let placed = base.with(Var::BombOrFlameAnimation, 18);
    let r = critique(&mut critic(), &step(&base, &placed, "BOMB N"));
    assert_eq!(names(&r), ["penalty-bomb-used"]);
    assert_eq!(value(&r, "penalty-bomb-used"), -REWARD_SMALL);
    let mut target = base.clone();
    target
        .enemy(3, enemy::STALFOS, 120, 121)
        .enemy(4, enemy::STALFOS, 130, 121);
    let placed = target.with(Var::BombOrFlameAnimation2, 19);
    let r = critique(
        &mut critic(),
        &hits(step(&target, &placed, "BOMB N"), &[(3, 1), (4, 1)]),
    );
    assert_eq!(
        value(&r, "reward-bomb-hit"),
        REWARD_MAXIMUM,
        "2 x 0.5, clamped"
    );
    assert!(r.contains("penalty-bomb-used"));
    // A bomb already ticking from an earlier step is not placed again.
    let ticking = base.with(Var::BombOrFlameAnimation, 18);
    assert!(critique(&mut critic(), &step(&ticking, &ticking, "BOMB N")).is_empty());
    // The bomb slot changing on a non-BOMB action is not a bomb use.
    assert!(critique(&mut critic(), &step(&base, &placed, "MOVE N 8")).is_empty());
}

// --- health ---

#[test]
fn health_changes() {
    let full = Ram::new();
    let hurt = full
        .with(Var::HeartsAndContainers, 0x20)
        .with(Var::PartialHearts, 0xFF); // 1 of 3
    // Healing is worth more the lower Link was.
    let r = critique(
        &mut critic(),
        &step(&hurt, &hurt.with(Var::HeartsAndContainers, 0x21), "WAIT 8"),
    );
    assert!(approx(
        value(&r, "reward-gained-health"),
        REWARD_SMALL + REWARD_MEDIUM * (2.0 / 3.0)
    ));
    let almost = full.with(Var::PartialHearts, 0x7F);
    let r = critique(&mut critic(), &step(&almost, &full, "WAIT 8"));
    assert!(approx(
        value(&r, "reward-gained-health"),
        REWARD_SMALL + REWARD_MEDIUM * (0.5 / 3.0)
    ));
    // `reward.fixed_pickup_reward`: the same pay however hurt Link was.
    let mut fixed = critic();
    fixed.fixed_pickup_reward = true;
    for (from, to) in [(&hurt, hurt.with(Var::HeartsAndContainers, 0x21)), (&almost, full.clone())] {
        let r = critique(&mut fixed, &step(from, &to, "WAIT 8"));
        assert!(approx(value(&r, "reward-gained-health"), REWARD_MAXIMUM));
    }

    // A heart container replaces the healing reward.
    let mut container = step(
        &full,
        &full.with(Var::HeartsAndContainers, 0x33),
        "MOVE N 8",
    );
    (container.health_gained, container.health_lost) = (1.0, 0.0);
    let r = critique(&mut critic(), &container);
    assert_eq!(names(&r), ["reward-gained-heart-container"]);

    // Damage: a quarter per half heart, clamped, plus losing the beams when Link was full.
    let r = critique(&mut critic(), &step(&full, &almost, "WAIT 8"));
    assert_eq!(names(&r), ["penalty-lost-health", "penalty-lost-beams"]);
    assert_eq!(value(&r, "penalty-lost-health"), -REWARD_SMALL);
    let r = critique(&mut critic(), &step(&almost, &hurt, "WAIT 8"));
    assert_eq!(
        names(&r),
        ["penalty-lost-health"],
        "not full before: no beams to lose"
    );
    assert_eq!(
        value(&r, "penalty-lost-health"),
        -REWARD_LARGE,
        "three half hearts"
    );
    let empty = full
        .with(Var::HeartsAndContainers, 0x20)
        .with(Var::PartialHearts, 0);
    let r = critique(&mut critic(), &step(&full, &empty, "WAIT 8"));
    assert_eq!(
        value(&r, "penalty-lost-health"),
        -REWARD_MAXIMUM,
        "six half hearts, clamped"
    );
    let no_sword = full.with(Var::Sword, 0);
    let r = critique(
        &mut critic(),
        &step(
            &no_sword,
            &no_sword.with(Var::PartialHearts, 0x7F),
            "WAIT 8",
        ),
    );
    assert_eq!(names(&r), ["penalty-lost-health"]);
}

/// Round 38: a Bubble's touch (asm:Bubble_Update: SwordBlockedLongTimer `$4C` := `$10`, one down every 10 frames)
/// costs `penalty-sword-blocked` once: the step the timer rises in, from 0 or reloaded while it ran down (a second
/// touch restarts the block); not the steps it runs down in. The red Bubble's flag (`$52E`, second quest) the same.
#[test]
fn a_bubble_touch_costs_once_per_touch() {
    let free = Ram::new();
    let timer = |t: u8| free.with(Var::SwordBlockedLongTimer, t);
    let paid = |r: &StepRewards| value(r, "penalty-sword-blocked");
    let r = critique(&mut critic(), &step(&free, &timer(0x10), "WAIT 8"));
    assert_eq!(names(&r), ["penalty-sword-blocked"], "no health lost: the touch's only term");
    assert_eq!(paid(&r), -env::config::SWORD_BLOCKED_PENALTY);
    for (before, after) in [(0x10, 0x10), (0x10, 0x0F), (0x01, 0x00), (0, 0)] {
        let r = critique(&mut critic(), &step(&timer(before), &timer(after), "WAIT 8"));
        assert!(names(&r).is_empty(), "{before:#x} -> {after:#x}: {:?}", names(&r));
    }
    let r = critique(&mut critic(), &step(&timer(0x0D), &timer(0x10), "WAIT 8"));
    assert_eq!(paid(&r), -env::config::SWORD_BLOCKED_PENALTY, "a second touch while blocked pays again");
    let r = critique(&mut critic(), &step(&free, &free.with(Var::SwordBlocked, 1), "WAIT 8"));
    assert_eq!(names(&r), ["penalty-sword-blocked"], "the red Bubble's block");
    // A touch and a hit in the same step: both terms; `reward.sword_blocked_penalty` sets the size, 0 drops it.
    let hurt = timer(0x10).with(Var::PartialHearts, 0x7F);
    let r = critique(&mut critic(), &step(&free, &hurt, "WAIT 8"));
    assert_eq!(names(&r), ["penalty-lost-health", "penalty-lost-beams", "penalty-sword-blocked"]);
    let mut off = critic();
    off.sword_blocked_penalty = 0.0;
    assert!(names(&critique(&mut off, &step(&free, &timer(0x10), "WAIT 8"))).is_empty());
}

#[test]
fn damage_halves_rewards_but_not_pickups() {
    let mut base = Ram::new();
    base.enemy(3, enemy::STALFOS, 40, 100);
    let curr = base.with(Var::PartialHearts, 0x7F).with(Var::Keys, 1);
    let r = critique(
        &mut critic(),
        &hits(step(&base, &curr, "SWORD N"), &[(3, 1)]),
    );
    assert_eq!(value(&r, "reward-hit"), REWARD_SMALL * 0.5);
    assert_eq!(value(&r, "reward-gained-keys"), REWARD_MAXIMUM);
}

#[test]
fn equipment_pickups_count_increases_only() {
    let base = Ram::new();
    let r = critique(
        &mut critic(),
        &step(
            &base,
            &base
                .with(Var::Rupees, 5)
                .with(Var::Compass, 0b1)
                .with(Var::RegularBoomerang, 1),
            "MOVE N 8",
        ),
    );
    assert_eq!(
        names(&r),
        [
            "reward-gained-boomerang",
            "reward-gained-compass",
            "reward-gained-rupees"
        ]
    );
    assert_eq!(value(&r, "reward-gained-rupees"), REWARD_LARGE, "5 rupees at once: a blue rupee's size");
    // Another level's compass bit, spending keys or rupees: nothing.
    let rich = base.with(Var::Rupees, 5).with(Var::Keys, 2);
    let r = critique(
        &mut critic(),
        &step(
            &rich,
            &rich
                .with(Var::Compass, 0b10)
                .with(Var::Keys, 1)
                .with(Var::Rupees, 0),
            "MOVE N 8",
        ),
    );
    assert!(r.is_empty(), "{:?}", names(&r));
}

/// Round 38 (the owner, 0:0x0f: "what should ACTUALLY be awarded for getting rupees is the 'rupees to add' counter
/// getting set once"): a pickup pays once on the step `InvRupeesToAdd` rises, sized by the amount, and nothing while
/// asm:Inv_ChangeRupees counts it into `InvRupees`.
#[test]
fn rupees_pay_once_per_pickup_not_per_counted_rupee() {
    let base = Ram::new().with(Var::Rupees, 76);
    let pickup = |amount: u8| {
        let r = critique(&mut critic(), &step(&base, &base.with(Var::RupeesToAdd, amount), "MOVE N 8"));
        assert_eq!(names(&r), ["reward-gained-rupees"]);
        value(&r, "reward-gained-rupees")
    };
    assert_eq!(pickup(1), REWARD_MEDIUM, "a rupee");
    assert_eq!(pickup(5), REWARD_LARGE, "a blue rupee");
    assert_eq!(pickup(30), REWARD_MAXIMUM, "a cave's gift");
    assert_eq!(pickup(100), REWARD_MAXIMUM);
    // A pickup counted in within the step (a rupee: 2 frames) pays the same.
    let r = critique(&mut critic(), &step(&base, &base.with(Var::Rupees, 77), "MOVE N 8"));
    assert_eq!(value(&r, "reward-gained-rupees"), REWARD_MEDIUM);
    // The count running: InvRupees up by what InvRupeesToAdd lost, nothing.
    let counting = base.with(Var::RupeesToAdd, 90);
    let counted = counting.with(Var::Rupees, 80).with(Var::RupeesToAdd, 86);
    assert!(critique(&mut critic(), &step(&counting, &counted, "WAIT 8")).is_empty());
    // A pickup while the count runs pays for the pickup alone.
    let picked = counted.with(Var::RupeesToAdd, 91);
    let r = critique(&mut critic(), &step(&counting, &picked, "MOVE N 8"));
    assert_eq!(value(&r, "reward-gained-rupees"), REWARD_LARGE, "4 counted, 5 picked up");
    // A price counted off, or an arrow's rupee: nothing.
    let buying = base.with(Var::RupeesToSubtract, 60);
    assert!(critique(&mut critic(), &step(&base, &buying, "MOVE N 8")).is_empty());
    let paying = buying.with(Var::Rupees, 70).with(Var::RupeesToSubtract, 54);
    assert!(critique(&mut critic(), &step(&buying, &paying, "WAIT 8")).is_empty());
}

/// F043: walking from the overworld into a dungeon whose map and compass Link already holds is no pickup; taking
/// them inside the dungeon still pays.
#[test]
fn entering_a_dungeon_with_its_map_is_no_pickup() {
    let held = Ram::new().with(Var::Map, 0b1).with(Var::Compass, 0b1);
    let outside = held.room(MapLocation::new(0, 0x37, false));
    let r = critique(&mut critic(), &step(&outside, &held, "MOVE N 8"));
    assert!(!names(&r).iter().any(|n| n.starts_with("reward-gained-")), "{:?}", names(&r));
    let base = Ram::new();
    let r = critique(&mut critic(), &step(&base, &base.with(Var::Map, 0b1), "MOVE N 8"));
    assert_eq!(value(&r, "reward-gained-map"), REWARD_MAXIMUM);
}

/// F092: a bomb pickup pays `reward-gained-bombs` (like rupees); placing a bomb pays nothing; the term is a known
/// reward term.
#[test]
fn bomb_pickups_pay_like_rupees() {
    let base = Ram::new().with(Var::Bombs, 2);
    let r = critique(&mut critic(), &step(&base, &base.with(Var::Bombs, 6), "MOVE N 8"));
    assert_eq!(names(&r), ["reward-gained-bombs"]);
    assert_eq!(value(&r, "reward-gained-bombs"), REWARD_MEDIUM);
    let r = critique(&mut critic(), &step(&base, &base.with(Var::Bombs, 1), "BOMB N"));
    assert!(!r.contains("reward-gained-bombs"));
    assert!(env::rewards::REWARD_TERMS.contains(&"reward-gained-bombs"));
}

// --- movement, rooms and stalling ---

#[test]
fn room_exits_reward_correct_new_rooms_once() {
    let base = Ram::new();
    let east = base.room(EAST);
    let mut critic = critic();
    let r = critique(&mut critic, &step(&base, &east, "MOVE E 8"));
    assert_eq!(names(&r), ["reward-new-location"]);
    let r = critique(&mut critic, &step(&base, &east, "MOVE E 8"));
    assert_eq!(names(&r), ["reward-revisit-location"]);
    assert_eq!(value(&r, "reward-revisit-location"), REWARD_TINY);
    let west = base.room(MapLocation::new(1, 0x43, false));
    let r = critique(&mut critic, &step(&base, &west, "MOVE W 8"));
    assert_eq!(names(&r), ["penalty-wrong-location"]);
    // Taking a triforce moves Link out of the room: no location reward or penalty.
    let r = critique(
        &mut critic,
        &step(&base, &west.with(Var::Triforce, 1), "MOVE W 8"),
    );
    assert_eq!(names(&r), ["reward-gained-triforce"]);
}

#[test]
fn stalling_ramps_after_the_grace_period_and_resets_on_progress() {
    let base = Ram::new();
    let mut critic = critic();
    let wait = step(&base, &base, "MOVE N 8");
    for _ in 0..ROOM_STEP_GRACE {
        assert!(critique(&mut critic, &wait).is_empty());
    }
    let first = value(&critique(&mut critic, &wait), "penalty-room-stalling");
    assert!(approx(
        first,
        -(ROOM_STEP_PENALTY_MIN + (ROOM_STEP_PENALTY_MAX - ROOM_STEP_PENALTY_MIN) / ROOM_STEP_RAMP)
    ));
    for _ in 0..ROOM_STEP_RAMP as u32 + 10 {
        critique(&mut critic, &wait);
    }
    assert_eq!(
        value(&critique(&mut critic, &wait), "penalty-room-stalling"),
        -ROOM_STEP_PENALTY_MAX
    );

    // A kill restarts the count (that step is step 1).
    let kill = step(&base, &base.with(Var::RoomKills, 1), "SWORD N");
    assert!(!critique(&mut critic, &kill).contains("penalty-room-stalling"));
    let mut stalled = 1;
    while !critique(&mut critic, &wait).contains("penalty-room-stalling") {
        stalled += 1;
    }
    assert_eq!(
        stalled, ROOM_STEP_GRACE,
        "the kill step is step 1; step 151 stalls"
    );
    // A pickup restarts it too; damage resets it without counting the step.
    let mut pickup = wait.clone();
    pickup.items_gained = vec![0x18];
    assert!(!critique(&mut critic, &pickup).contains("penalty-room-stalling"));
    for _ in 1..ROOM_STEP_GRACE {
        critique(&mut critic, &wait);
    }
    let hurt = step(&base, &base.with(Var::PartialHearts, 0x7F), "WAIT 8");
    assert!(!critique(&mut critic, &hurt).contains("penalty-room-stalling"));
    for _ in 0..ROOM_STEP_GRACE {
        assert!(!critique(&mut critic, &wait).contains("penalty-room-stalling"));
    }
    assert!(critique(&mut critic, &wait).contains("penalty-room-stalling"));
}

/// A Vire's Red Keese (or a Gleeok head) dies without the game counting it in RoomKills ($34F): the observed kill
/// still restarts the stall count.
#[test]
fn an_uncounted_kill_restarts_the_stall_count() {
    let mut base = Ram::new();
    base.enemy(3, enemy::RED_KEESE, 40, 100);
    let mut critic = critic();
    let wait = step(&base, &base, "MOVE N 8");
    for _ in 0..ROOM_STEP_GRACE + 10 {
        critique(&mut critic, &wait);
    }
    assert!(critique(&mut critic, &wait).contains("penalty-room-stalling"));
    let mut emptied = base.clone();
    emptied.0[tables::OBJ_ID.0 + 3] = 0;
    let mut kill = step(&base, &emptied, "SWORD N");
    kill.observed.enemies_killed = vec![3];
    assert_eq!(kill.state.room_kills(), kill.previous.room_kills());
    assert!(!critique(&mut critic, &kill).contains("penalty-room-stalling"));
    let quiet = step(&emptied, &emptied, "MOVE N 8");
    assert!(!critique(&mut critic, &quiet).contains("penalty-room-stalling"));
}

#[test]
fn quiet_enemy_penalty_needs_a_live_active_enemy_past_the_grace() {
    let mut base = Ram::new();
    base.enemy(3, enemy::RED_GORIYA, 40, 100);
    let wait = step(&base, &base, "WAIT 8");
    let mut quiet = GameplayCritic::new(false, 1.0, 0.03);
    let r = critique_with(
        &mut quiet,
        &wait,
        &objective(&[EAST]),
        None,
        ROOM_STEP_GRACE + 1,
    );
    assert_eq!(value(&r, "penalty-quiet-enemies"), -0.03);
    let r = critique_with(
        &mut quiet,
        &wait,
        &objective(&[EAST]),
        None,
        ROOM_STEP_GRACE,
    );
    assert!(!r.contains("penalty-quiet-enemies"));
    // Off by default, and not for a spawning (inactive) or dying enemy.
    let r = critique_with(&mut critic(), &wait, &objective(&[EAST]), None, 1000);
    assert!(!r.contains("penalty-quiet-enemies"));
    let mut spawning = base.clone();
    spawning.0[tables::OBJ_SPAWN_STATE.0 + 3] = 1;
    let r = critique_with(
        &mut quiet,
        &step(&spawning, &spawning, "WAIT 8"),
        &objective(&[EAST]),
        None,
        1000,
    );
    assert!(!r.contains("penalty-quiet-enemies"));
    // F122: a living 0-HP monster (Gels, Keese have ObjHP 0) still counts; one in the death sparkle doesn't.
    let mut keese = Ram::new();
    keese.enemy(3, enemy::BLUE_KEESE, 40, 100);
    keese.0[tables::OBJ_HEALTH.0 + 3] = 0;
    let r = critique_with(
        &mut quiet,
        &step(&keese, &keese, "WAIT 8"),
        &objective(&[EAST]),
        None,
        1000,
    );
    assert_eq!(value(&r, "penalty-quiet-enemies"), -0.03);
    let mut dying = keese.clone();
    dying.0[tables::OBJ_SPAWN_STATE.0 + 3] = 0x10;
    let r = critique_with(
        &mut quiet,
        &step(&dying, &dying, "WAIT 8"),
        &objective(&[EAST]),
        None,
        1000,
    );
    assert!(!r.contains("penalty-quiet-enemies"));
}

#[test]
fn pbrs_rewards_wavefront_progress_from_every_step() {
    let base = Ram::new(); // tile (15, 10)
    let wf =
        Wavefront::from_distances([((15, 10), 10), ((15, 9), 9), ((15, 8), 8), ((15, 11), 11)]);
    let mut critic = GameplayCritic::new(true, 1.0, 0.0);
    let obj = objective(&[EAST]);
    let run = |critic: &mut GameplayCritic,
               from: &Ram,
               to: &Ram,
               action: &str,
               wf: Option<&Wavefront>| {
        critique_with(critic, &step(from, to, action), &obj, wf, 0)
    };
    let up = base.link(120, 133);
    // The first step of an episode pays like any other.
    let r = run(&mut critic, &base, &up, "MOVE N 8", Some(&wf));
    assert_eq!(value(&r, "reward-pbrs-movement"), 1.0 / 20.0);
    critic.clear();
    // A step that takes damage (its knockback) or changes rooms is not shaped; the next step pays from where it
    // left Link.
    let hurt = up.with(Var::PartialHearts, 0x7F);
    assert!(!run(&mut critic, &base, &hurt, "MOVE N 8", Some(&wf)).contains("reward-pbrs-movement"));
    let r = run(&mut critic, &hurt, &hurt.link(120, 125), "MOVE N 8", Some(&wf));
    assert_eq!(value(&r, "reward-pbrs-movement"), 1.0 / 20.0);
    assert!(!run(&mut critic, &base, &up.room(EAST), "MOVE N 8", Some(&wf)).contains("reward-pbrs-movement"));
    let r = run(&mut critic, &base, &up, "MOVE N 8", Some(&wf));
    assert_eq!(value(&r, "reward-pbrs-movement"), 1.0 / 20.0);
    let down = base.link(120, 149);
    let r = run(&mut critic, &up, &down, "MOVE S 16", Some(&wf));
    assert_eq!(value(&r, "penalty-pbrs-movement"), -2.0 / 20.0);
    // Standing still on a non-MOVE step is not shaped; moving into an unreachable tile isn't either.
    assert!(run(&mut critic, &down, &down, "WAIT 8", Some(&wf)).is_empty());
    let off = base.link(160, 149);
    assert!(run(&mut critic, &down, &off, "MOVE E 16", Some(&wf)).is_empty());
    // Without a wavefront nothing is shaped.
    assert!(run(&mut critic, &off, &up, "MOVE W 16", None).is_empty());
    // PBRS off: never.
    assert!(
        run(
            &mut GameplayCritic::new(false, 1.0, 0.0),
            &base,
            &up,
            "MOVE N 8",
            Some(&wf)
        )
        .is_empty()
    );
}

#[test]
fn wallmaster_rooms_discourage_the_walls() {
    let mut base = Ram::new();
    base.enemy(5, enemy::WALLMASTER, 200, 100);
    let center = base.link(64, 141); // tile (8, 10)
    let wall = base.link(32, 141); // tile (4, 10): on the west wall band
    let obj = objective(&[EAST]);
    let r = critique_with(
        &mut critic(),
        &step(&center, &wall, "MOVE W 16"),
        &obj,
        None,
        0,
    );
    assert_eq!(names(&r), ["penalty-moved-onto-wallmaster"]);
    assert_eq!(value(&r, "penalty-moved-onto-wallmaster"), -REWARD_TINY);
    let r = critique_with(&mut critic(), &step(&wall, &wall, "SWORD N"), &obj, None, 0);
    assert!(r.contains("penalty-fighting-wallmaster"));
    let r = critique_with(
        &mut critic(),
        &step(&wall, &center, "MOVE E 16"),
        &obj,
        None,
        0,
    );
    assert_eq!(names(&r), ["reward-moved-off-wallmaster"]);
    assert!(approx(
        value(&r, "reward-moved-off-wallmaster"),
        REWARD_TINY - REWARD_MINIMUM
    ));
    // The wall tile is the objective (the exit): no penalty.
    let exit = Objective {
        targets: [Target::Tile((4, 10))].into_iter().collect(),
        ..objective(&[EAST])
    };
    let r = critique_with(
        &mut critic(),
        &step(&center, &wall, "MOVE W 16"),
        &exit,
        None,
        0,
    );
    assert!(r.is_empty(), "{:?}", names(&r));
    // No Wallmaster: walls are fine.
    let plain = Ram::new();
    let r = critique_with(
        &mut critic(),
        &step(&plain.link(64, 141), &plain.link(32, 141), "MOVE W 16"),
        &obj,
        None,
        0,
    );
    assert!(r.is_empty());
}

#[test]
fn opening_a_bomb_wall_pays_once() {
    let Some(nes) = common::open_core() else {
        return;
    };
    let mut s = zelda::Session::new(nes, MAP.clone());
    let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(0);
    s.load(
        &common::load_named_state("1_44w_0").unwrap(),
        &Default::default(),
        &mut rng,
    )
    .unwrap();
    let closed = Ram(*s.state().ram());
    // $EE: doors opened in this visit (S = bit 2).
    let mut open = closed.clone();
    open.0[0xEE] |= 0x04;
    let mut critic = critic();
    let r = critique(&mut critic, &step(&closed, &open, "MOVE S 8"));
    assert_eq!(value(&r, "reward-opened-bomb-wall"), REWARD_LARGE);
    assert!(
        !critique(&mut critic, &step(&closed, &open, "MOVE S 8"))
            .contains("reward-opened-bomb-wall")
    );
    // A new episode may earn it again.
    critic.clear();
    assert!(
        critique(&mut critic, &step(&closed, &open, "MOVE S 8"))
            .contains("reward-opened-bomb-wall")
    );
}

#[test]
fn unknown_reward_profile_is_a_config_error() {
    let mut config = SpecialistConfig::from_yaml(
        "name: t\nrooms: ['(1, 0x44)']\nsuccess: [exit-correct]\n",
        "t.yaml",
        None,
    )
    .unwrap();
    config.reward_profile = "fancy".into();
    let err = GameplayCritic::for_config(&config).unwrap_err();
    assert_eq!(
        err.to_string(),
        "specialists/t.yaml: reward_profile: unknown \"fancy\"; known: [\"default\"]"
    );
}

// --- end conditions ---

fn ends(
    conditions: &mut EndConditions,
    change: &Change,
    next_rooms: &[MapLocation],
    tail: u32,
) -> Option<String> {
    let objective = objective(next_rooms);
    conditions.check(&StepView {
        change,
        prev_objective: &objective,
        map: &MAP,
        linger_tail_steps: tail,
    })
}

fn conditions(success: &[&str], one_room: bool, no_progress: Option<u32>) -> EndConditions {
    let success: Vec<String> = success.iter().map(|s| s.to_string()).collect();
    EndConditions::new(&[ROOM, EAST], &success, one_room, no_progress, None, 2000)
}

#[test]
fn endings_fire_in_priority_order() {
    let base = Ram::new();
    let all = [TRIFORCE_COLLECTED, BOSS_DEAD_AND_HEART, EXIT_CORRECT];
    // Dying while the room changes, the triforce is taken and the heart container appears: death first.
    let far = base.room(MapLocation::new(1, 0x22, false));
    let everything = far
        .with(Var::Mode, 17)
        .with(Var::Triforce, 1)
        .with(Var::HeartsAndContainers, 0x33);
    let mut wm = base.clone();
    wm.enemy(5, enemy::WALLMASTER, 200, 100);
    let mut c = conditions(&all, false, None);
    assert_eq!(
        ends(&mut c, &step(&wm, &everything, "WAIT 8"), &[], 0).as_deref(),
        Some(DEATH)
    );
    // Alive: the triforce exempts the jump from being a Wallmaster grab.
    let alive = everything.with(Var::Mode, 5);
    assert_eq!(
        ends(&mut c, &step(&wm, &alive, "WAIT 8"), &[], 0).as_deref(),
        Some(TRIFORCE_COLLECTED)
    );
    let no_triforce = alive.with(Var::Triforce, 0);
    assert_eq!(
        ends(&mut c, &step(&wm, &no_triforce, "WAIT 8"), &[], 0).as_deref(),
        Some(WALLMASTERED)
    );
    assert_eq!(
        ends(&mut c, &step(&base, &no_triforce, "WAIT 8"), &[], 0).as_deref(),
        Some(BOSS_DEAD_AND_HEART)
    );
    // Without those successes listed, the same step is just a wrong exit.
    let mut plain = conditions(&[EXIT_CORRECT], false, None);
    assert_eq!(
        ends(&mut plain, &step(&base, &alive, "WAIT 8"), &[], 0).as_deref(),
        Some(EXIT_WRONG)
    );
}

#[test]
fn wallmaster_grab_needs_a_wallmaster_and_a_jump() {
    let mut wm = Ram::new();
    wm.enemy(5, enemy::WALLMASTER, 200, 100);
    let far = MapLocation::new(1, 0x73, false);
    let mut c = conditions(&[EXIT_CORRECT], true, None);
    // One room over through a door is an exit, not a grab.
    let next = step(&wm, &wm.room(EAST), "MOVE E 8");
    assert_eq!(
        ends(&mut c, &next, &[EAST], 0).as_deref(),
        Some(EXIT_CORRECT)
    );
    assert_eq!(
        ends(&mut c, &step(&wm, &wm.room(far), "WAIT 8"), &[], 0).as_deref(),
        Some(WALLMASTERED)
    );
    // The same jump without a Wallmaster in the room before is a plain wrong exit.
    let plain = Ram::new();
    assert_eq!(
        ends(&mut c, &step(&plain, &wm.room(far), "WAIT 8"), &[], 0).as_deref(),
        Some(EXIT_WRONG)
    );
}

#[test]
fn room_exits_follow_the_family_and_enter_room_successes() {
    let base = Ram::new();
    let east = base.room(EAST);
    let mut c = conditions(&[EXIT_CORRECT], false, None);
    // Correct exit into another room of the family: the episode goes on.
    assert_eq!(
        ends(&mut c, &step(&base, &east, "MOVE E 8"), &[EAST], 0),
        None
    );
    let mut one = conditions(&[EXIT_CORRECT], true, None);
    assert_eq!(
        ends(&mut one, &step(&base, &east, "MOVE E 8"), &[EAST], 0).as_deref(),
        Some(EXIT_CORRECT)
    );
    let west = base.room(MapLocation::new(1, 0x43, false));
    assert_eq!(
        ends(&mut c, &step(&base, &west, "MOVE W 8"), &[EAST], 0).as_deref(),
        Some(EXIT_WRONG)
    );
    // An enter-room success names the room, even inside the family.
    let target = enter_room_name(EAST);
    let mut enter = conditions(&[&target], false, None);
    assert_eq!(
        ends(&mut enter, &step(&base, &east, "MOVE E 8"), &[EAST], 0),
        Some(target)
    );
    // Leaving the family through a correct exit is exit-correct.
    let out = MapLocation::new(1, 0x34, false);
    assert_eq!(
        ends(&mut c, &step(&base, &base.room(out), "MOVE N 8"), &[out], 0).as_deref(),
        Some(EXIT_CORRECT)
    );
}

#[test]
fn timeout_catches_stuck_link_and_rooms_without_progress() {
    let base = Ram::new();
    let still = step(&base, &base, "WAIT 8");
    let mut c = conditions(&[EXIT_CORRECT], false, None);
    for _ in 1..50 {
        assert_eq!(ends(&mut c, &still, &[EAST], 0), None);
    }
    assert_eq!(ends(&mut c, &still, &[EAST], 0).as_deref(), Some(STUCK));
    // Hitting something while standing still is not being stuck; clear() starts over.
    c.clear();
    let hitting = hits(still.clone(), &[(3, 1)]);
    for _ in 0..100 {
        assert_eq!(ends(&mut c, &hitting, &[EAST], 0), None);
    }
    c.clear();
    // Moving around one room for more than 2000 steps is no progress.
    let (a, b) = (base.link(120, 133), base.link(120, 141));
    let (there, back) = (step(&a, &b, "MOVE S 8"), step(&b, &a, "MOVE N 8"));
    let mut ending = None;
    let mut n = 0;
    while ending.is_none() {
        n += 1;
        ending = ends(&mut c, if n % 2 == 0 { &there } else { &back }, &[EAST], 0);
        assert!(n <= 2001, "never ended");
    }
    assert_eq!((ending.as_deref(), n), (Some(NO_PROGRESS), 2001));
}

/// Round 28: decisions made or ended while a Like-Like holds Link (`IsLinkParalyzed` `$512`) are not counted toward
/// the position limit (he cannot move until it dies): a hold of any length is not `stuck`, the count from before the
/// hold is kept (not begun anew), and standing still after it counts on.
#[test]
fn a_hold_does_not_count_toward_stuck() {
    let base = Ram::new();
    let mut held = base.clone();
    held.0[0x512] = 1;
    let (still, caught, holding, freed) =
        (step(&base, &base, "WAIT 8"), step(&base, &held, "WAIT 8"), step(&held, &held, "SWORD N"), step(&held, &base, "SWORD N"));
    let mut c = conditions(&[EXIT_CORRECT], false, None);
    for _ in 0..30 {
        assert_eq!(ends(&mut c, &still, &[EAST], 0), None);
    }
    assert_eq!(ends(&mut c, &caught, &[EAST], 0), None);
    for _ in 0..500 {
        assert_eq!(ends(&mut c, &holding, &[EAST], 0), None);
    }
    assert_eq!(ends(&mut c, &freed, &[EAST], 0), None);
    // 30 counted before the hold: the 20th still decision after it is the 50th.
    for _ in 1..20 {
        assert_eq!(ends(&mut c, &still, &[EAST], 0), None);
    }
    assert_eq!(ends(&mut c, &still, &[EAST], 0).as_deref(), Some(STUCK));
}

/// F078: the no-new-room limit follows the specialist's max_steps: with max_steps 3000 a one_room episode moving
/// in its room has no ending through step 3000 (the env's `max-steps` truncation comes first).
#[test]
fn the_no_progress_limit_follows_max_steps() {
    let base = Ram::new();
    let (a, b) = (base.link(120, 133), base.link(120, 141));
    let (there, back) = (step(&a, &b, "MOVE S 8"), step(&b, &a, "MOVE N 8"));
    let success = vec![EXIT_CORRECT.to_owned()];
    let mut c = EndConditions::new(&[ROOM, EAST], &success, true, None, None, 3000);
    for n in 1..=3000 {
        assert_eq!(ends(&mut c, if n % 2 == 0 { &there } else { &back }, &[EAST], 0), None, "step {n}");
    }
}

#[test]
fn no_room_progress_replaces_the_timeout() {
    let base = Ram::new();
    let still = step(&base, &base, "WAIT 8");
    let mut c = conditions(&[EXIT_CORRECT], false, Some(30));
    for _ in 0..100 {
        assert_eq!(
            ends(&mut c, &still, &[EAST], 29),
            None,
            "no stuck timeout either"
        );
    }
    assert_eq!(
        ends(&mut c, &still, &[EAST], 30).as_deref(),
        Some(NO_ROOM_PROGRESS)
    );
}

#[test]
fn no_damage_ends_attack_loops_that_never_connect() {
    let base = Ram::new();
    let swing = step(&base, &base, "SWORD E");
    let success = vec![EXIT_CORRECT.to_owned()];
    let mut c = EndConditions::new(&[ROOM, EAST], &success, false, Some(1000), Some(5), 2000);
    let swings = |c: &mut EndConditions, n: usize| {
        (0..n)
            .map(|_| ends(c, &swing, &[EAST], 0))
            .collect::<Vec<_>>()
    };
    assert!(swings(&mut c, 4).iter().all(Option::is_none));
    // Damage dealt, damage taken and a new room each restart the count; a swing that misses doesn't.
    for reset in [
        hits(step(&base, &base, "SWORD E"), &[(1, 1)]),
        step(&base, &base.with(Var::HeartsAndContainers, 0x21), "WAIT 8"),
        step(&base, &base.room(EAST), "MOVE E 8"),
    ] {
        assert_eq!(ends(&mut c, &reset, &[EAST], 0), None);
        assert!(swings(&mut c, 4).iter().all(Option::is_none));
    }
    assert_eq!(swings(&mut c, 1), [Some(NO_DAMAGE.to_owned())]);
    // Endings count observed damage only: a look-ahead's credit on the firing step doesn't restart the count.
    c.clear();
    assert!(swings(&mut c, 4).iter().all(Option::is_none));
    let fired = predicted(step(&base, &base, "SWORD E"), &[(1, 1)]);
    assert_eq!(ends(&mut c, &fired, &[EAST], 0).as_deref(), Some(NO_DAMAGE));
    c.clear();
    assert!(swings(&mut c, 4).iter().all(Option::is_none), "reset starts over");
}

#[test]
fn classification_errors_say_what_to_fix() {
    let success = vec![EXIT_CORRECT.to_owned()];
    let mut endings = IndexMap::new();
    endings.insert(MAX_STEPS.to_owned(), "great".to_owned());
    let err = classify_ending(MAX_STEPS, &success, &endings).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Bad outcome \"great\" for ending \"max-steps\""
    );
    endings.clear();
    endings.insert(WALLMASTERED.to_owned(), HARD_FAILURE.to_owned());
    let err = classify_ending("enter-room:1:0x45", &success, &endings).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.starts_with("Ending \"enter-room:1:0x45\" is not classified"),
        "{msg}"
    );
    assert!(
        msg.contains("exit-correct") && msg.contains("list it in the specialist's success/endings"),
        "{msg}"
    );
    // A specialist override wins over the central table.
    endings.insert(DEATH.to_owned(), "soft-failure".to_owned());
    assert_eq!(
        classify_ending(DEATH, &success, &endings),
        Ok("soft-failure")
    );
}

// --- reward.heart_value (round 25) ---

/// `goriya` (whose rooms hold 1:0x44) with `reward.heart_value` set to `exit`, `potential`, `low_health`.
fn heart_config(exit: bool, potential: f64, low_health: f64) -> SpecialistConfig {
    let mut config = (*common::game_data().specialist("goriya").unwrap()).clone();
    config.reward.heart_value = env::config::HeartValueConfig { exit, potential, low_health };
    config
}

/// d2first's step 60 is 1:0x44 W -> E (goriya), step 61 1:0x45 W -> N (wallmaster) (64 and 65 before round 31's
/// Dungeon 1).
fn heart_context(values: &env::heart_value::HeartValues, next: Option<usize>) -> HeartContext<'_> {
    HeartContext { values: Some(values), step: Some(60), next, clips: true }
}

fn score(config: &SpecialistConfig, change: &Change, heart: &HeartContext) -> env::rewards::Scored {
    let objective = objective(&[EAST]);
    let mut critic = GameplayCritic::for_config(config).unwrap();
    let mut ends = EndConditions::new(
        &config.rooms,
        &config.success,
        config.one_room,
        None,
        None,
        config.max_steps as u32,
    );
    let c = Critique {
        change,
        map: &MAP,
        prev_objective: &objective,
        objective: &objective,
        prev_wavefront: None,
        quiet_steps: 0,
    };
    score_step(config, &mut critic, &mut ends, &c, 1.0, 1, &change.state, heart).unwrap()
}

/// Off (the default) is today's reward exactly: the table and the route steps change nothing.
#[test]
fn heart_value_off_is_todays_reward() {
    let data = common::game_data();
    let values = data.heart_values.as_deref().expect("triforce/heart_value.json ships with the route");
    let config = heart_config(false, 0.0, 0.0);
    assert_eq!(config.reward.heart_value, env::config::HeartValueConfig::default());
    assert!(data.heart_values_for(&config, "d2first").is_none(), "off reads no table");
    let full = Ram::new();
    let hurt = full.with(Var::HeartsAndContainers, 0x21).with(Var::PartialHearts, 0x7F); // 1.5 of 3
    let pickup = step(&hurt, &hurt.with(Var::PartialHearts, 0xFF), "WAIT 8");
    let hit = step(&full, &hurt, "WAIT 8");
    let exit = step(&hurt, &hurt.room(EAST), "MOVE E 8");
    for change in [&pickup, &hit, &exit] {
        let without = score(&config, change, &HeartContext::default());
        let with = score(&config, change, &heart_context(values, Some(61)));
        assert_eq!(without.rewards.terms().collect::<Vec<_>>(), with.rewards.terms().collect::<Vec<_>>());
        assert_eq!(without.rewards.value().to_bits(), with.rewards.value().to_bits());
        assert_eq!((without.ending, without.outcome), (with.ending, with.outcome));
    }
    let scored = score(&config, &exit, &heart_context(values, Some(61)));
    assert_eq!(scored.outcome, Some("success"));
    let terminal = scored.rewards.get("reward-terminal-success").unwrap().value;
    assert!(approx(terminal, env::rewards::success_reward_value(1.5, 3.0, 0.5)), "{terminal}");
}

/// `potential`: a change of hearts pays scale × (V(step, after) − V(step, before)), outside the step's clamp;
/// `low_health`: the hit penalty grows with the hearts lost; `exit`: the success terminal's health part is
/// V(next step, hearts) / V(next step, max).
#[test]
fn heart_value_terms() {
    let data = common::game_data();
    let values = data.heart_values.as_deref().unwrap();
    let curve = |step: usize| values.curve(Some(step), true).unwrap();
    let full = Ram::new();
    let low = full.with(Var::HeartsAndContainers, 0x20).with(Var::PartialHearts, 0x7F); // 0.5 of 3
    let one = full.with(Var::HeartsAndContainers, 0x20).with(Var::PartialHearts, 0xFF); // 1 of 3
    let heart = heart_context(values, Some(61));

    // A hit from 1 to 0.5 hearts at 1:0x44: the flat −0.25, then 20 × ΔV unclamped.
    let config = heart_config(false, 20.0, 0.0);
    let hit = step(&one, &low, "WAIT 8");
    let r = score(&config, &hit, &heart).rewards;
    let dv = curve(60).at(0.5) - curve(60).at(1.0);
    assert!(dv < -0.1, "a heart at one heart is worth much: {dv}");
    assert!(approx(value(&r, "penalty-heart-value"), 20.0 * dv));
    assert!(approx(r.value(), -REWARD_SMALL + 20.0 * dv), "the term is outside the clamp: {}", r.value());
    // The pickup back pays the same.
    let r = score(&config, &step(&low, &one, "WAIT 8"), &heart).rewards;
    assert!(approx(value(&r, "reward-heart-value"), -20.0 * dv));
    // Off the route or without a curve: no term.
    let r = score(&config, &hit, &HeartContext { step: None, ..heart }).rewards;
    assert!(!r.contains("penalty-heart-value"));

    // low_health 2: the hit's penalty × (1 + 2 × (1 − 0.5 / 3)).
    let config = heart_config(false, 0.0, 2.0);
    let r = score(&config, &hit, &heart).rewards;
    assert!(approx(value(&r, "penalty-heart-value"), -REWARD_SMALL * 2.0 * (1.0 - 0.5 / 3.0)));

    // exit: 20 × (0.5 + 0.5 × V(61, 1) / V(61, top)) instead of 20 × (0.5 + 0.5 × 1/3).
    let config = heart_config(true, 0.0, 0.0);
    let exit = step(&one, &one.room(EAST), "MOVE E 8");
    let r = score(&config, &exit, &heart).rewards;
    let priced = 20.0 * (0.5 + 0.5 * curve(61).at(1.0) / curve(61).top());
    assert!(approx(value(&r, "reward-terminal-success"), priced));
    // No next step (off the route): the linear fraction.
    let r = score(&config, &exit, &HeartContext { next: None, ..heart }).rewards;
    assert!(approx(value(&r, "reward-terminal-success"), env::rewards::success_reward_value(1.0, 3.0, 0.5)));
}

/// The shipped table matches the route it names; an edited route is refused by the families that read it, and the
/// curve interpolates between half hearts.  Dungeon 7's and Dungeon 6's steps have curves (round 26's and round 27's
/// chains), Dungeon 8's (round 28's) and level 9's (round 29's); since 1.9.0.0 `d2first` runs to the end of the game
/// and the table is its own, 354 rows since round 31.
#[test]
fn heart_value_table_checks_the_route() {
    let data = common::game_data();
    let values = data.heart_values.as_deref().unwrap();
    assert_eq!(values.route, "d2first");
    let steps = &data.map.routes["d2first"];
    let visits = &data.map.visits["d2first"];
    values.check(steps, visits).unwrap();
    let mut moved = steps.clone();
    moved[64].exit = "N".into();
    let err = values.check(&moved, visits).unwrap_err().0;
    assert!(err.contains("step 64") && err.contains("refit"), "{err}");
    assert!(values.check(&steps[1..], &visits[1..]).is_err());
    assert!(values.check(&steps[..297], &visits[..297]).is_err(), "fewer steps");
    assert_eq!(values.routes, ["d2first"]);
    let curve = values.curve(Some(64), true).unwrap();
    assert!(approx(curve.at(0.75), (curve.values[1] + curve.values[2]) / 2.0));
    assert_eq!(curve.at(99.0), curve.top());
    assert_eq!(curve.at(0.0), 0.0);
    // Steps on one way only have no curve on the other (0x31 E -> N is the clip way's).
    let clip_only = steps.iter().position(|s| s.way == Some(zelda::game_map::Way::Clips)).unwrap();
    assert!(values.curve(Some(clip_only), false).is_none() && values.curve(Some(clip_only), true).is_some());
    // Dungeon 7's 0x1b from W (step 236), Dungeon 6's 0x38 (step 266), Dungeon 8's 0x3e (step 287) and level 9's
    // 0x10 (step 336, the silver arrow's Wizzrobes) are priced.
    for (k, room) in [(236, "7:0x1b"), (266, "6:0x38"), (287, "8:0x3e"), (336, "9:0x10")] {
        assert_eq!(values.steps[k].room, room);
        assert!(values.curve(Some(k), true).is_some_and(|c| c.top() > c.at(6.0)), "{room}");
    }
    let mut config = heart_config(true, 0.0, 0.0);
    data.check_heart_values(&config).unwrap();
    assert!(data.heart_values_for(&config, "d2first").is_some());
    assert!(data.heart_values_for(&config, "clips-gauntlet").is_none());
    config.route = Some("clips-gauntlet".into());
    assert!(data.check_heart_values(&config).is_err(), "a table for another route");
}

// --- round 34: a truncation the config names a failure in `endings` ends the episode as that failure ---

fn wait_scored(config: &SpecialistConfig, no_damage: Option<u32>, steps: u64) -> env::rewards::Scored {
    let full = Ram::new();
    let change = step(&full, &full, "WAIT 8");
    let objective = objective(&[EAST]);
    let mut critic = GameplayCritic::for_config(config).unwrap();
    let mut ends = EndConditions::new(&config.rooms, &config.success, config.one_room, None, no_damage, 2000);
    let c = Critique {
        change: &change,
        map: &MAP,
        prev_objective: &objective,
        objective: &objective,
        prev_wavefront: None,
        quiet_steps: 0,
    };
    score_step(config, &mut critic, &mut ends, &c, 1.0, steps, &change.state, &HeartContext::default()).unwrap()
}

/// By default `max-steps` and `no-damage` truncate (no terminal reward, a soft failure); named `hard-failure` in
/// `endings` they terminate with the terminal failure penalty, as a death does.
#[test]
fn a_truncation_named_a_failure_ends_as_one() {
    use env::rewards::{PENALTY_TERMINAL_FAILURE, SOFT_FAILURE, TERMINAL_FAILURE_PENALTY_VALUE};
    let mut config = (*common::game_data().specialist("goriya").unwrap()).clone();
    config.max_steps = 5;
    let cases = [(None, 5, MAX_STEPS), (Some(1), 1, NO_DAMAGE)];
    for (no_damage, steps, ending) in cases {
        let s = wait_scored(&config, no_damage, steps);
        assert_eq!(s.ending.as_deref(), Some(ending));
        assert!(s.truncated && !s.terminated, "{ending}");
        assert_eq!(s.outcome, Some(SOFT_FAILURE));
        assert!(s.rewards.get(PENALTY_TERMINAL_FAILURE).is_none());
    }
    config.endings = IndexMap::from([
        (MAX_STEPS.to_owned(), HARD_FAILURE.to_owned()),
        (NO_DAMAGE.to_owned(), HARD_FAILURE.to_owned()),
    ]);
    for (no_damage, steps, ending) in cases {
        let s = wait_scored(&config, no_damage, steps);
        assert_eq!(s.ending.as_deref(), Some(ending));
        assert!(s.terminated && !s.truncated, "{ending}");
        assert_eq!(s.outcome, Some(HARD_FAILURE));
        let penalty = s.rewards.get(PENALTY_TERMINAL_FAILURE).unwrap().value;
        assert!(approx(penalty, -TERMINAL_FAILURE_PENALTY_VALUE), "{penalty}");
    }
    assert!(wait_scored(&config, None, 1).ending.is_none(), "before the budget nothing ends");
}
