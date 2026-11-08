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
use env::rewards::{
    HARD_FAILURE, REWARD_LARGE, REWARD_MAXIMUM, REWARD_MEDIUM, REWARD_MINIMUM, REWARD_SMALL,
    REWARD_TINY, StepRewards,
};
use indexmap::IndexMap;
use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};
use zelda::enums::enemy;
use zelda::objectives::{Objective, ObjectiveKind};
use zelda::room::{Target, Wavefront};
use zelda::tables::{self, RAM_SIZE};
use zelda::{Action, ActionMask, Change, GameMap, GameView, MapLocation, Var};

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
        interrupt: None,
        control_frames: 8,
        projectile_blocks: Vec::new(),
        previous,
        state,
        action_mask: ActionMask::default(),
        health_lost: (-delta).max(0.0),
        health_gained: delta.max(0.0),
        enemies_hit: Vec::new(),
        enemies_stunned: Vec::new(),
        items_gained: Vec::new(),
    }
}

fn hits(mut change: Change, hits: &[(u8, i32)]) -> Change {
    change.enemies_hit = hits.to_vec();
    change
}

fn objective(next_rooms: &[MapLocation]) -> Objective {
    Objective {
        kind: ObjectiveKind::Move,
        targets: BTreeSet::new(),
        next_rooms: next_rooms.iter().copied().collect(),
        pbrs_targets: BTreeSet::new(),
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
    for action in ["MOVE N 8", "WAIT 8", "BOMB N"] {
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

#[test]
fn dodongo_stun_pays_and_a_swallowed_bomb_costs() {
    let mut walking = Ram::new();
    walking.enemy(1, enemy::DODONGO2, 112, 141);
    let with_status = |status: u8| {
        let mut ram = walking.clone();
        ram.0[tables::OBJ_STATUS.0 + 1] = status;
        ram
    };
    // The stun usually lands a step after the bomb (48-frame fuse): whatever the action, walking -> 2 pays.
    let r = critique(&mut critic(), &step(&walking, &with_status(2), "MOVE N 8"));
    assert_eq!(value(&r, "reward-boss-stunned"), REWARD_LARGE);
    let r = critique(&mut critic(), &step(&walking, &with_status(1), "WAIT 8"));
    assert_eq!(value(&r, "penalty-boss-ate-bomb"), -REWARD_MEDIUM);
    // Staying stunned, or recovering, is neither.
    let r = critique(&mut critic(), &step(&with_status(2), &with_status(2), "WAIT 8"));
    assert!(!names(&r).iter().any(|n| n.contains("boss")), "{:?}", names(&r));
    let r = critique(&mut critic(), &step(&with_status(2), &walking, "WAIT 8"));
    assert!(!names(&r).iter().any(|n| n.contains("boss")), "{:?}", names(&r));
}

#[test]
fn boss_transitions_count_each_dodongo() {
    let mut walking = Ram::new();
    walking.enemy(1, enemy::DODONGO, 112, 141).enemy(2, enemy::DODONGO2, 64, 141);
    let mut after = walking.clone();
    after.0[tables::OBJ_STATUS.0 + 1] = 2;
    after.0[tables::OBJ_STATUS.0 + 2] = 1;
    let change = step(&walking, &after, "WAIT 8");
    assert_eq!(BossTransitions::of(&change), BossTransitions { stuns: 1, eats: 1 });
    let r = critique(&mut critic(), &change);
    assert_eq!(value(&r, "reward-boss-stunned"), REWARD_LARGE);
    assert_eq!(value(&r, "penalty-boss-ate-bomb"), -REWARD_MEDIUM);
    assert_eq!(BossTransitions::of(&step(&after, &walking, "WAIT 8")), BossTransitions::default());
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
    assert_eq!(names(&far), ["reward-beam-hit", "reward-beam-distance"]);
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
    assert_eq!(value(&r, "reward-gained-rupees"), REWARD_MEDIUM);
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
    // Off by default, and not for a spawning (inactive) or zero-health enemy.
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
    let mut empty = base.clone();
    empty.0[tables::OBJ_HEALTH.0 + 3] = 0;
    let r = critique_with(
        &mut quiet,
        &step(&empty, &empty, "WAIT 8"),
        &objective(&[EAST]),
        None,
        1000,
    );
    assert!(!r.contains("penalty-quiet-enemies"));
}

#[test]
fn pbrs_rewards_wavefront_progress_from_the_last_baseline() {
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
    // The first step only sets the baseline.
    assert!(run(&mut critic, &base, &base, "MOVE N 8", Some(&wf)).is_empty());
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
    EndConditions::new(&[ROOM, EAST], &success, one_room, no_progress, None)
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
    let mut c = EndConditions::new(&[ROOM, EAST], &success, false, Some(1000), Some(5));
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
