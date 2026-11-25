//! Gameplay critic: per-step rewards (profile `default`).
//!
//! Term names, magnitudes, clamps and the order the critiques run in are those of the old Python critic: combat
//! decay per room, health halving of positive rewards, the room-stall ramp, Wallmaster shaping, one-time puzzle
//! rewards and PBRS on the previous state's wavefront.

use crate::config::{ConfigError, SpecialistConfig};
use crate::end_conditions::RETREAT;
use crate::rewards::{
    NEUTRAL, Outcome, REWARD_LARGE, REWARD_MAXIMUM, REWARD_MEDIUM, REWARD_MINIMUM, REWARD_SMALL,
    REWARD_TINY, StepRewards,
};
use std::collections::{HashMap, HashSet};
use zelda::enums::{
    ActionKind, AnimationKind, AnimationState, Direction, MapLocation, SwordKind, Tile, enemy,
};
use zelda::game::GameView;
use zelda::objectives::{FLAG_ITEM_TAKEN, Objective, exit_is_correct};
use zelda::room::{Target, Wavefront};
use zelda::room_state::{SecretTrigger, decode_room_state, room_flags};
use zelda::{Change, GameMap, Var, beams, tables};

pub const BEAM_DISTANCE_THRESHOLD: f32 = 48.0;
pub const PBRS_SCALE: f64 = 20.0;
/// Steps in a room before the stalling penalty starts.
pub const ROOM_STEP_GRACE: u32 = 150;
pub const ROOM_STEP_PENALTY_MIN: f64 = 0.01;
pub const ROOM_STEP_PENALTY_MAX: f64 = 0.02;
/// Steps over the grace period to reach the maximum penalty.
pub const ROOM_STEP_RAMP: f64 = 1850.0;
/// Combat rewards decay after this many events per room (no farming respawning enemies).
pub const COMBAT_DECAY_THRESHOLD: i32 = 8;
pub const COMBAT_DECAY_RATE: f64 = 0.5;

/// Reward profiles `reward_profile` may name.
pub const REWARD_PROFILES: [&str; 1] = ["default"];
/// ObjDir of the push block (object slot $B): the direction it was pushed (asm:PushBlock_State0Idle).
const PUSH_BLOCK_DIRECTION: u16 = 0x98 + 0x0B;
/// asm:FluteUsedState: 0 until the recorder is played in the room (the game zeroes it on room entry).
const FLUTE_USED_STATE: u16 = 0x51B;

/// Equipment pickups, in `_EQUIPMENT_ATTRS` order: (name, value). Bombs pay like rupees (a dropped resource); the
/// cheat ledger's grants happen outside the action's `Change`, so a grant is no pickup. Rupees pay once per pickup by
/// its size ([`rupees_posted`], [`rupee_reward`]), not this table's value.
const EQUIPMENT: [(&str, &str, f64); 21] = [
    ("sword", "reward-gained-sword", REWARD_MAXIMUM),
    ("arrows", "reward-gained-arrows", REWARD_MAXIMUM),
    ("bow", "reward-gained-bow", REWARD_MAXIMUM),
    ("candle", "reward-gained-candle", REWARD_MAXIMUM),
    ("whistle", "reward-gained-whistle", REWARD_MAXIMUM),
    ("food", "reward-gained-food", REWARD_MAXIMUM),
    ("potion", "reward-gained-potion", REWARD_MAXIMUM),
    ("magic_rod", "reward-gained-magic_rod", REWARD_MAXIMUM),
    ("raft", "reward-gained-raft", REWARD_MAXIMUM),
    ("book", "reward-gained-book", REWARD_MAXIMUM),
    ("ring", "reward-gained-ring", REWARD_MAXIMUM),
    ("ladder", "reward-gained-ladder", REWARD_MAXIMUM),
    ("magic_key", "reward-gained-magic_key", REWARD_MAXIMUM),
    (
        "power_bracelet",
        "reward-gained-power_bracelet",
        REWARD_MAXIMUM,
    ),
    ("letter", "reward-gained-letter", REWARD_MAXIMUM),
    ("boomerang", "reward-gained-boomerang", REWARD_MAXIMUM),
    ("compass", "reward-gained-compass", REWARD_MAXIMUM),
    ("map", "reward-gained-map", REWARD_MAXIMUM),
    ("rupees", "reward-gained-rupees", REWARD_MEDIUM),
    ("keys", "reward-gained-keys", REWARD_MAXIMUM),
    ("bombs", "reward-gained-bombs", REWARD_MEDIUM),
];

/// The value `link.<attr>` compares by (enum value, bool as int, raw count).
fn equipment_value(state: &GameView, attr: &str) -> i32 {
    let byte = |v: Var| state.peek(v.addr()) as i32;
    match attr {
        "sword" => state.sword() as i32,
        "arrows" => byte(Var::Arrows),
        "bow" => byte(Var::Bow),
        "candle" => byte(Var::Candle),
        "whistle" => byte(Var::Whistle),
        "food" => byte(Var::Food),
        "potion" => byte(Var::Potion),
        "magic_rod" => byte(Var::MagicRod),
        "raft" => byte(Var::Raft),
        "book" => byte(Var::Book),
        "ring" => byte(Var::Ring),
        "ladder" => byte(Var::StepLadder),
        "magic_key" => byte(Var::MagicKey),
        "power_bracelet" => byte(Var::PowerBracelet),
        "letter" => byte(Var::Letter),
        "boomerang" => state.boomerang() as i32,
        "compass" => state.has_compass() as i32,
        "map" => state.has_map() as i32,
        "rupees" => rupees_posted(state),
        "keys" => state.keys(),
        "bombs" => state.bombs(),
        _ => unreachable!("equipment {attr}"),
    }
}

/// Rupees Link holds or is owed: `InvRupees` plus what is still to be counted in, minus what is still to be counted
/// off. A pickup (asm:Item_TakeOneRupee, five of them for the blue rupee) or a cave's gift (asm:Inv_PostRupeeCredit)
/// raises `InvRupeesToAdd` once; asm:Inv_ChangeRupees then moves 1 a second frame from it into `InvRupees`, which
/// leaves this sum alone. So it rises on the step of the pickup by the amount, and not while the counter runs
/// (round 38: Link waited in 0:0x0f's cave for the 100 rupees to count, paid at every step they did).
fn rupees_posted(state: &GameView) -> i32 {
    state.rupees() + state.var(Var::RupeesToAdd) - state.var(Var::RupeesToSubtract)
}

/// `reward-gained-rupees` for a pickup of `amount` rupees: a rupee `REWARD_MEDIUM` (what one paid before round 38),
/// a blue rupee (5) `REWARD_LARGE`, 10 or more (a cave's gift, a stash) `REWARD_MAXIMUM`.
pub fn rupee_reward(amount: i32) -> f64 {
    match amount {
        ..5 => REWARD_MEDIUM,
        5..10 => REWARD_LARGE,
        _ => REWARD_MAXIMUM,
    }
}

/// A step's Dodongo events: `stuns`, a bomb's smoke stunned him (ObjState $AC+slot entered 2; the sword can kill),
/// `eats`, he swallowed a bomb (Dodongo_ObjBombHits $437+slot went up). Read from what changed, not from the
/// state pair: one step can hold a swallow, the bloat and a stun (1 → 0 → 2), or a stun ending and a new one
/// (the stun timer restarts at $20 and so goes *up*). The critic rewards/penalises them; the envs total them
/// per episode, visit and chain room (`boss: {stuns, eats}`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BossTransitions {
    pub stuns: u32,
    pub eats: u32,
}

impl BossTransitions {
    pub fn of(change: &Change) -> Self {
        let (prev, curr) = (&change.previous, &change.state);
        let mut t = Self::default();
        for before in prev.enemies.iter().filter(|e| matches!(e.id, enemy::DODONGO | enemy::DODONGO2)) {
            let Some(after) = curr.enemy_by_index(before.index).filter(|e| e.id == before.id) else {
                continue;
            };
            // The frame the smoke stuns him the timer is still 0; the next update sets it to $20 (not a new stun).
            let restunned = before.stun_timer > 0 && after.stun_timer > before.stun_timer;
            if after.status == 2 && (before.status != 2 || restunned) {
                t.stuns += 1;
            }
            let hits = |state: &GameView| state.ram()[tables::DODONGO_BOMB_HITS.0 + before.index as usize];
            t.eats += hits(curr).saturating_sub(hits(prev)) as u32;
        }
        t
    }
}

impl std::ops::AddAssign for BossTransitions {
    fn add_assign(&mut self, other: Self) {
        self.stuns += other.stuns;
        self.eats += other.eats;
    }
}

/// What the critic sees after one step.
pub struct Critique<'a> {
    pub change: &'a Change,
    pub map: &'a GameMap,
    /// The objective of `change.previous` (which exits are correct).
    pub prev_objective: &'a Objective,
    /// The objective of `change.state` (Wallmaster objective tiles).
    pub objective: &'a Objective,
    /// The PBRS wavefront of `change.previous` (None with PBRS off).
    pub prev_wavefront: Option<&'a Wavefront>,
    /// The shared linger tracker's tail steps after this step.
    pub quiet_steps: u32,
}

/// The default reward profile (`GameplayCritic`).
#[derive(Clone, Debug)]
pub struct GameplayCritic {
    pub pbrs: bool,
    pub block_reward: f64,
    pub quiet_enemy_penalty: f64,
    /// `reward.fixed_pickup_reward`: health gained pays `REWARD_MAXIMUM` instead of the urgency scale.
    pub fixed_pickup_reward: bool,
    /// `reward.boss_ate_bomb_penalty`: `penalty-boss-ate-bomb` per swallowed bomb (positive magnitude).
    pub boss_ate_bomb_penalty: f64,
    /// `reward.left_drop_penalty`: `penalty-left-drop` per wanted drop left on the floor (positive magnitude; 0 off).
    pub left_drop_penalty: f64,
    /// `reward.sword_blocked_penalty`: `penalty-sword-blocked` per step a Bubble's touch landed in (positive
    /// magnitude; 0 off).
    pub sword_blocked_penalty: f64,
    /// `reward.beam_kill_reward`, `reward.beam_timed_reward`, `reward.beam_unfired_penalty` (round 38; 0 off each):
    /// [`GameplayCritic::critique_beams`].
    pub beam_kill_reward: f64,
    pub beam_timed_reward: f64,
    pub beam_unfired_penalty: f64,
    /// `reward.bomb_catch_min` (round 40; 0 off): [`GameplayCritic::critique_item_usage`].
    pub bomb_catch_min: u8,
    /// The `retreat` ending is neutral (the default; `endings: {retreat: ...}` can make it a failure): leaving through
    /// a route step's `retreat` exit then costs no `penalty-wrong-location` either.
    pub neutral_retreat: bool,
    correct_locations: HashSet<MapLocation>,
    /// The maze steps (room, `MazeStep` after the loop) correct maze loops reached, rewarded like new rooms.
    correct_loops: HashSet<(MapLocation, u8)>,
    room_combat_counts: HashMap<MapLocation, i32>,
    room_steps: u32,
    stunned_enemies: HashSet<(MapLocation, u8)>,
    puzzles_rewarded: HashSet<(MapLocation, String)>,
    /// `reward-beam-timed` payments per room, decayed as combat rewards are ([`COMBAT_DECAY_THRESHOLD`]).
    timed_shots: HashMap<MapLocation, i32>,
}

impl GameplayCritic {
    pub fn new(pbrs: bool, block_reward: f64, quiet_enemy_penalty: f64) -> GameplayCritic {
        GameplayCritic {
            pbrs,
            block_reward,
            quiet_enemy_penalty,
            fixed_pickup_reward: false,
            boss_ate_bomb_penalty: REWARD_MEDIUM,
            left_drop_penalty: 0.0,
            sword_blocked_penalty: crate::config::SWORD_BLOCKED_PENALTY,
            beam_kill_reward: crate::config::BEAM_KILL_REWARD,
            beam_timed_reward: crate::config::BEAM_TIMED_REWARD,
            beam_unfired_penalty: crate::config::BEAM_UNFIRED_PENALTY,
            bomb_catch_min: 0,
            neutral_retreat: true,
            correct_locations: HashSet::new(),
            correct_loops: HashSet::new(),
            room_combat_counts: HashMap::new(),
            room_steps: 0,
            stunned_enemies: HashSet::new(),
            puzzles_rewarded: HashSet::new(),
            timed_shots: HashMap::new(),
        }
    }

    /// The configured critic (`create_critic`); errors for an unknown `reward_profile`.
    pub fn for_config(config: &SpecialistConfig) -> Result<GameplayCritic, ConfigError> {
        if !REWARD_PROFILES.contains(&config.reward_profile.as_str()) {
            return Err(ConfigError(format!(
                "specialists/{}.yaml: reward_profile: unknown {:?}; known: {REWARD_PROFILES:?}",
                config.name, config.reward_profile
            )));
        }
        let mut critic = GameplayCritic::new(
            config.reward.pbrs,
            config.reward.block_reward,
            config.reward.quiet_enemy_penalty,
        );
        critic.fixed_pickup_reward = config.reward.fixed_pickup_reward;
        critic.boss_ate_bomb_penalty = config.reward.boss_ate_bomb_penalty;
        critic.left_drop_penalty = config.reward.left_drop_penalty;
        critic.sword_blocked_penalty = config.reward.sword_blocked_penalty;
        critic.beam_kill_reward = config.reward.beam_kill_reward;
        critic.beam_timed_reward = config.reward.beam_timed_reward;
        critic.beam_unfired_penalty = config.reward.beam_unfired_penalty;
        critic.bomb_catch_min = config.reward.bomb_catch_min;
        critic.neutral_retreat = config.endings.get(RETREAT).is_none_or(|outcome| outcome == NEUTRAL);
        Ok(critic)
    }

    /// Called at every episode start.
    pub fn clear(&mut self) {
        self.correct_locations.clear();
        self.correct_loops.clear();
        self.room_combat_counts.clear();
        self.room_steps = 0;
        self.stunned_enemies.clear();
        self.puzzles_rewarded.clear();
        self.timed_shots.clear();
    }

    /// Decay multiplier for combat rewards in a room (counts the event).
    fn combat_decay(&mut self, location: MapLocation) -> f64 {
        Self::decay(&mut self.room_combat_counts, location)
    }

    /// The decay of the next event `counts` holds for `location` (counts it): 1 up to [`COMBAT_DECAY_THRESHOLD`]
    /// events, then halving.
    fn decay(counts: &mut HashMap<MapLocation, i32>, location: MapLocation) -> f64 {
        let count = counts.entry(location).or_insert(0);
        let before = *count;
        *count += 1;
        Self::decay_after(before)
    }

    fn decay_after(events: i32) -> f64 {
        let excess = events - COMBAT_DECAY_THRESHOLD;
        if excess <= 0 { 1.0 } else { COMBAT_DECAY_RATE.powi(excess) }
    }

    /// Adds this step's gameplay rewards (`critique_gameplay`).
    pub fn critique(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        if change.gained_triforce() {
            rewards.add(Outcome::reward("reward-gained-triforce", REWARD_MAXIMUM));
        }
        self.critique_attack(c, rewards);
        self.critique_beams(c, rewards);
        self.critique_dodongo(c, rewards);
        self.critique_recorder(c, rewards);
        self.critique_item_usage(c, rewards);
        self.critique_equipment_pickup(c, rewards);
        self.critique_puzzles(c, rewards);
        self.critique_location_change(c, rewards);
        self.critique_movement(c, rewards);
        self.critique_block(c, rewards);
        if self.quiet_enemy_penalty != 0.0
            && c.quiet_steps > ROOM_STEP_GRACE
            && change
                .state
                .enemies
                .iter()
                .any(|e| e.is_active() && !e.is_dying())
        {
            rewards.add(Outcome::penalty(
                "penalty-quiet-enemies",
                -self.quiet_enemy_penalty,
            ));
        }
        self.critique_health_change(c, rewards);
        // Round 38: a Bubble's touch costs no health but blocks the sword for 151-160 frames (asm:Bubble_Update) and
        // shoves Link; one payment a touch (the block timer's rise: a second touch while blocked restarts it, and
        // pays again), flat, whatever the hearts.
        if self.sword_blocked_penalty > 0.0 && change.bubble_touched() {
            rewards.add(Outcome::penalty("penalty-sword-blocked", -self.sword_blocked_penalty));
        }
        self.critique_wallmaster(c, rewards);
        if change.health_lost > 0.0 {
            rewards.scale_rewards(0.5, &["reward-gained-"]);
        }
    }

    fn critique_equipment_pickup(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let (prev, curr) = (&c.change.previous, &c.change.state);
        // A treasure the route step refuses (routes.yaml `treasure: refuse`, Dungeon 5's 0x65 westbound): its pickup
        // (the room's item-taken flag rising, asm:Item_TryTake) costs instead of paying.
        let room = curr.full_location();
        if c.prev_objective.treasure_refused
            && room == prev.full_location()
            && room.level > 0
            && !room.in_cave
            && room_flags(curr.ram(), room.value) & FLAG_ITEM_TAKEN != 0
            && room_flags(prev.ram(), room.value) & FLAG_ITEM_TAKEN == 0
        {
            rewards.add(Outcome::penalty("penalty-refused-treasure", -REWARD_LARGE));
            return;
        }
        for (attr, name, value) in EQUIPMENT {
            // The map and compass are the current level's: walking into a dungeon whose map Link holds is no pickup.
            if matches!(attr, "map" | "compass") && prev.level() != curr.level() {
                continue;
            }
            let (before, after) = (equipment_value(prev, attr), equipment_value(curr, attr));
            if before < after {
                let value = if attr == "rupees" { rupee_reward(after - before) } else { value };
                rewards.add(Outcome::reward(name, value));
            }
        }
    }

    fn critique_puzzles(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let (prev, curr) = (&c.change.previous, &c.change.state);
        if prev.full_location() != curr.full_location() || curr.level() == 0 {
            return;
        }
        let before = decode_room_state(prev.ram());
        let after = decode_room_state(curr.ram());
        let location = curr.full_location();
        for (direction, door) in &after.doors {
            let Some(old) = before.door(*direction) else {
                continue;
            };
            if old.is_open || !door.is_open {
                continue;
            }
            match door.kind() {
                "key" => self.reward_puzzle(
                    location,
                    format!("door-{}", direction.name()),
                    Outcome::reward("reward-opened-locked-door", REWARD_MEDIUM),
                    rewards,
                ),
                "bomb" => self.reward_puzzle(
                    location,
                    format!("bomb-{}", direction.name()),
                    Outcome::reward("reward-opened-bomb-wall", REWARD_LARGE),
                    rewards,
                ),
                _ => {}
            }
        }
        // A block that opens a door or stairs, or the one a game.yaml `push_block` room needs pushed one of its ways
        // (Dungeon 1's 0x22: the stairs are there from the start, walled in by blocks).
        let pushed = Direction::from_value(curr.peek(PUSH_BLOCK_DIRECTION));
        let route_block = curr
            .game_map()
            .get(location)
            .is_some_and(|r| pushed.is_some_and(|d| r.push_block.contains(&d)));
        if after.has_push_block
            && before.block_push_complete == 0
            && after.block_push_complete > 0
            && (route_block
                || matches!(
                    after.secret_trigger,
                    SecretTrigger::BlockDoor | SecretTrigger::BlockStairs
                ))
        {
            self.reward_puzzle(
                location,
                "push-block".into(),
                Outcome::reward("reward-pushed-block", REWARD_LARGE),
                rewards,
            );
        }
    }

    fn reward_puzzle(
        &mut self,
        location: MapLocation,
        kind: String,
        reward: Outcome,
        rewards: &mut StepRewards,
    ) {
        let key = (location, kind);
        if self.puzzles_rewarded.contains(&key) || rewards.contains(reward.name) {
            return;
        }
        self.puzzles_rewarded.insert(key);
        rewards.add(reward);
    }

    fn critique_health_change(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        let (prev, curr) = (&change.previous, &change.state);
        if prev.max_health() < curr.max_health() {
            rewards.add(Outcome::reward(
                "reward-gained-heart-container",
                REWARD_MAXIMUM,
            ));
        } else if change.health_gained != 0.0 {
            if !change.gained_triforce() {
                let value = if self.fixed_pickup_reward {
                    REWARD_MAXIMUM
                } else {
                    let urgency = 1.0 - prev.health() / curr.max_health() as f64;
                    REWARD_SMALL + REWARD_MEDIUM * urgency
                };
                rewards.add(Outcome::reward("reward-gained-health", value));
            }
        } else if change.health_lost != 0.0 {
            let half_hearts = change.health_lost * 2.0;
            let penalty = (REWARD_SMALL * half_hearts).min(REWARD_MAXIMUM);
            rewards.add(Outcome::penalty("penalty-lost-health", -penalty));
            if prev.is_health_full() && !curr.is_health_full() && curr.sword() != SwordKind::None {
                rewards.add(Outcome::penalty("penalty-lost-beams", -REWARD_SMALL));
            }
        }
    }

    fn critique_wallmaster(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        let (prev, curr) = (&change.previous, &change.state);
        if prev.full_location() != curr.full_location()
            || !curr.enemies.iter().any(|e| e.id == enemy::WALLMASTER)
        {
            return;
        }
        let tile = curr.link.tile();
        let objective_tile = c.objective.targets.contains(&Target::Tile(tile))
            || c.objective.pbrs_targets.contains(&Target::Tile(tile));
        let on_wall = |t: Tile| t.0 == 0x4 || t.0 == 0x1a || t.1 == 0x4 || t.1 == 0x10;
        let curr_on = on_wall(tile) && !objective_tile;
        let prev_on = on_wall(prev.link.tile()) && !objective_tile;
        let moving = change.action.kind == ActionKind::Move;
        if curr_on {
            if !moving {
                rewards.add(Outcome::penalty(
                    "penalty-fighting-wallmaster",
                    -REWARD_TINY,
                ));
            } else {
                rewards.add(Outcome::penalty(
                    "penalty-moved-onto-wallmaster",
                    -REWARD_TINY,
                ));
            }
        } else if prev_on && moving {
            rewards.add(Outcome::reward(
                "reward-moved-off-wallmaster",
                REWARD_TINY - REWARD_MINIMUM,
            ));
        }
    }

    fn critique_block(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let blocks = &c.change.projectile_blocks;
        if blocks.is_empty() {
            return;
        }
        let mut value = 0.0;
        for event in blocks {
            value += REWARD_MEDIUM * self.combat_decay(event.full_location) * self.block_reward;
        }
        if value != 0.0 {
            rewards.add(Outcome {
                name: "reward-block-projectile",
                value,
                count: blocks.len() as u32,
            });
        }
    }

    /// `reward.bomb_catch_min` (round 40, Darknut40's switched-off proposal): the step placed a bomb that hurt fewer
    /// monsters than the minimum, where one bomb is not right (no boss hit, not the 10th damage-free kill's bomb, not
    /// a `requires_bombs` room). Such a bomb pays no hit reward (`reward-hit`, `reward-bomb-hit`); its cost stands.
    fn short_catch(&self, c: &Critique) -> bool {
        let change = c.change;
        let prev = &change.previous;
        self.bomb_catch_min > 0
            && change.placed_bomb()
            && change.hits() < usize::from(self.bomb_catch_min)
            && !prev.next_kill_arms_drop()
            && !c.map.requires_bombs(prev.full_location())
            && !change.enemies_hit.iter().any(|&(slot, _)| prev.enemy_by_index(slot).is_some_and(|e| enemy::is_boss(e.id)))
    }

    fn critique_attack(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        let (prev, curr) = (&change.previous, &change.state);
        let kind = change.action.kind;
        let is_boss = |index: u8| {
            prev.enemy_by_index(index)
                .or_else(|| curr.enemy_by_index(index))
                .is_some_and(|e| enemy::is_boss(e.id))
        };
        let boss_hits: i32 = change
            .enemies_hit
            .iter()
            .filter(|(i, _)| is_boss(*i))
            .map(|(_, n)| n)
            .sum();
        let hits = change.hits();
        if boss_hits != 0 {
            rewards.add_scaled(
                Outcome::reward("reward-boss-hit", REWARD_LARGE),
                boss_hits as f64,
            );
        } else if hits != 0
            && prev.are_beams_available()
            && curr.animation_state(AnimationKind::Beams) != AnimationState::Inactive
        {
            let decay = self.combat_decay(curr.full_location());
            rewards.add(Outcome::reward("reward-beam-hit", REWARD_MEDIUM * decay));
            for (index, _) in &change.enemies_hit {
                if curr
                    .enemy_by_index(*index)
                    .is_some_and(|e| e.distance > BEAM_DISTANCE_THRESHOLD)
                {
                    rewards.add(Outcome::reward("reward-beam-distance", REWARD_TINY * decay));
                    break;
                }
            }
        } else if hits != 0 && !self.short_catch(c) {
            if !curr.in_cave() {
                let decay = self.combat_decay(curr.full_location());
                rewards.add(Outcome::reward("reward-hit", REWARD_SMALL * decay));
            } else {
                rewards.add(Outcome::penalty("penalty-attack-cave", -REWARD_MAXIMUM));
            }
        } else if !matches!(
            kind,
            // No strike of their own: BOMB and CANDLE hit later (the fire moves for 31 frames), RECORDER and FOOD never.
            ActionKind::Move
                | ActionKind::Wait
                | ActionKind::Bomb
                | ActionKind::Clip
                | ActionKind::Recorder
                | ActionKind::Food
                | ActionKind::Candle
        ) && change.enemies_stunned.is_empty()
        {
            rewards.add(Outcome::penalty("penalty-attack-miss", -REWARD_MINIMUM));
        }

        if !change.enemies_stunned.is_empty() && kind == ActionKind::Boomerang {
            let location = curr.full_location();
            let mut any_new = false;
            for &i in &change.enemies_stunned {
                if self.stunned_enemies.insert((location, i)) {
                    any_new = true;
                }
            }
            if any_new {
                let decay = self.combat_decay(location);
                rewards.add(Outcome::reward(
                    "reward-boomerang-stun",
                    REWARD_SMALL * decay,
                ));
            }
        }
    }

    /// Round 38, sword shots at range ([`zelda::beams`]; the owner: "The model is super happy to walk up to enemies
    /// and just hit them up close. It should get more reward for long range kills (out of melee range), and possibly
    /// a small penalty for ignoring a lined up shot"). On a step that fires a shot ([`beams::fired`]):
    /// `reward-beam-kill` when the hits the look-ahead credits to it kill a monster (not a boss) that stood out of
    /// melee range ([`beams::MELEE_REACH`]), decayed with the room's combat rewards; `reward-beam-timed` when the
    /// decision's projection ([`beams::lined_up`]) had the shot meet a monster in the direction fired, whether it
    /// does or not (decayed on its own count per room). On a MOVE or WAIT from a decision lined up so (the shot not
    /// taken; another weapon used is a choice, not an oversight): `penalty-beam-unfired`, flat.
    fn critique_beams(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        if change.entered_room() {
            return;
        }
        let prev = &change.previous;
        let fired = beams::fired(change);
        let location = prev.full_location();
        if fired && self.beam_kill_reward > 0.0 {
            let far_kill = change.enemies_hit.iter().any(|&(slot, damage)| {
                prev.enemy_by_index(slot).is_some_and(|e| {
                    !enemy::is_boss(e.id) && e.distance > beams::MELEE_REACH && damage >= e.health.max(1)
                })
            });
            if far_kill {
                // `reward-beam-hit` counted this step's event already.
                let events = self.room_combat_counts.get(&location).copied().unwrap_or(1);
                let decay = Self::decay_after(events - 1);
                rewards.add(Outcome::reward("reward-beam-kill", self.beam_kill_reward * decay));
            }
        }
        if self.beam_timed_reward <= 0.0 && self.beam_unfired_penalty <= 0.0 {
            return;
        }
        let lined = beams::lined_up(prev);
        if lined.is_empty() {
            return;
        }
        if !fired {
            if self.beam_unfired_penalty > 0.0 && matches!(change.action.kind, ActionKind::Move | ActionKind::Wait) {
                rewards.add(Outcome::penalty("penalty-beam-unfired", -self.beam_unfired_penalty));
            }
        } else if self.beam_timed_reward > 0.0 && lined.contains(&change.action.direction) {
            let decay = Self::decay(&mut self.timed_shots, location);
            rewards.add(Outcome::reward("reward-beam-timed", self.beam_timed_reward * decay));
        }
    }

    /// Bombs: placing one costs (`penalty-bomb-used`: a bomb that hits nothing is a loss), each hit the look-ahead
    /// credits to the placing step pays (`reward-bomb-hit`), and a 10th damage-free kill made with a bomb, which
    /// arms a forced drop of 4 bombs, pays `reward-armed-bomb-drop` on the step it happens (the owner, Nov 11; picking
    /// the bombs up is `reward-gained-bombs`). With `reward.bomb_catch_min` the hits of a bomb that hurt fewer pay
    /// nothing ([`GameplayCritic::short_catch`]).
    fn critique_item_usage(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        if change.armed_bomb_drop() {
            rewards.add(Outcome::reward("reward-armed-bomb-drop", REWARD_LARGE));
        }
        if change.placed_bomb() {
            rewards.add(Outcome::penalty("penalty-bomb-used", -REWARD_SMALL));
            if change.hits() != 0 && !self.short_catch(c) {
                rewards.add_scaled(Outcome::reward("reward-bomb-hit", REWARD_MEDIUM), change.hits() as f64);
            }
        }
    }

    /// Dodongo (round 12): only the sword hurts him, and only while a bomb's smoke has him stunned (ObjState 2);
    /// a bomb he swallows instead (ObjState 1) wastes a bomb and the turn. Rewards the stun, penalises the
    /// swallow ([`BossTransitions`]); the kill itself is `reward-boss-hit` and the heart container ends the episode.
    fn critique_dodongo(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let t = BossTransitions::of(c.change);
        if t.stuns > 0 {
            rewards.add(Outcome::reward("reward-boss-stunned", REWARD_LARGE));
        }
        if t.eats > 0 && self.boss_ate_bomb_penalty > 0.0 {
            rewards.add(Outcome::penalty("penalty-boss-ate-bomb", -self.boss_ate_bomb_penalty));
        }
    }

    /// Digdogger (round 24): the recorder played where a big Digdogger hasn't heard it (asm:FluteUsedState $51B
    /// rising from 0; the action mask offers RECORDER only there, `zelda::actions::recorder_wanted`) starts his split
    /// into the small one, the only form a weapon hurts. Pays `reward-recorder-split` once per room per episode; the
    /// small one's hits are boss hits (`enemy::BOSS_KINDS`), and the recorder is never an attack miss.
    fn critique_recorder(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        let (prev, curr) = (&change.previous, &change.state);
        let location = curr.full_location();
        if change.action.kind == ActionKind::Recorder
            && location == prev.full_location()
            && prev.peek(FLUTE_USED_STATE) == 0
            && curr.peek(FLUTE_USED_STATE) != 0
            && prev.enemies.iter().any(|e| e.is_big_digdogger())
            && self.puzzles_rewarded.insert((location, "recorder-split".to_owned()))
        {
            rewards.add(Outcome::reward("reward-recorder-split", REWARD_LARGE));
        }
    }

    /// A correct exit pays `reward-new-location` the first time into a room, else `reward-revisit-location`; a wrong
    /// one `penalty-wrong-location` (but a neutral retreat). A maze loop is a room transition (round 24): a correct
    /// one (`MazeLoop::correct`) pays as a new room once per step of the sequence, a wrong one as a wrong exit.
    fn critique_location_change(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        if let Some(l) = change.maze_loop() {
            if !l.correct(&c.prev_objective.next_rooms) {
                rewards.add(Outcome::penalty("penalty-wrong-location", -REWARD_SMALL));
            } else if self.correct_loops.insert((l.room, l.step)) {
                rewards.add(Outcome::reward("reward-new-location", REWARD_MAXIMUM));
            } else {
                rewards.add(Outcome::reward("reward-revisit-location", REWARD_TINY));
            }
            return;
        }
        let (prev, curr) = (
            change.previous.full_location(),
            change.state.full_location(),
        );
        if prev == curr || change.gained_triforce() {
            return;
        }
        // Leaving wanted drops behind (round 25: Manhandla's bombs): those not about to expire (16+ lifetime ticks,
        // about 32 frames).
        if self.left_drop_penalty > 0.0 {
            let left = change
                .previous
                .items
                .iter()
                .filter(|it| it.timer >= 16 && crate::objective::wanted_drop(&change.previous, c.prev_objective.own_bombs, it.id))
                .count();
            if left > 0 {
                rewards.add(Outcome::penalty("penalty-left-drop", -self.left_drop_penalty * left as f64));
            }
        }
        // Round 43: leaving a carrier alive where the step's carriers come first pays no exit (`carrier-left`).
        if exit_is_correct(c.map, &change.previous, c.prev_objective, curr) && !c.prev_objective.carriers_first() {
            if self.correct_locations.contains(&curr) {
                rewards.add(Outcome::reward("reward-revisit-location", REWARD_TINY));
            } else {
                rewards.add(Outcome::reward("reward-new-location", REWARD_MAXIMUM));
                self.correct_locations.insert(curr);
            }
        } else if !(self.neutral_retreat && c.prev_objective.retreats.contains(&curr)) {
            rewards.add(Outcome::penalty("penalty-wrong-location", -REWARD_SMALL));
        }
    }

    /// Room stall penalty plus PBRS: F = (d(prev tile) − d(curr tile)) / PBRS_SCALE on the previous state's
    /// wavefront; MOVE always, other kinds only when Link's tile changed. Not on a step that enters a room (a maze
    /// loop too: it puts Link back at the far edge, no fall in potential) or takes damage (no cross-room distances,
    /// no credit for a knockback); the step after one pays from where it left Link.
    fn critique_movement(&mut self, c: &Critique, rewards: &mut StepRewards) {
        let change = c.change;
        let (prev, curr) = (&change.previous, &change.state);
        if change.health_lost != 0.0 || change.entered_room() {
            self.room_steps = 0;
            return;
        }
        // A kill counts even when the game's RoomKills ($34F) doesn't (a Vire's Red Keese, Gleeok's heads).
        if !change.items_gained.is_empty()
            || curr.room_kills() > prev.room_kills()
            || !change.observed.enemies_killed.is_empty()
        {
            self.room_steps = 0;
        }
        self.room_steps += 1;
        if self.room_steps > ROOM_STEP_GRACE {
            let excess = (self.room_steps - ROOM_STEP_GRACE) as f64;
            let t = (excess / ROOM_STEP_RAMP).min(1.0);
            let penalty =
                -(ROOM_STEP_PENALTY_MIN + t * (ROOM_STEP_PENALTY_MAX - ROOM_STEP_PENALTY_MIN));
            rewards.add(Outcome::penalty("penalty-room-stalling", penalty));
        }
        if !self.pbrs {
            return;
        }
        let tile = curr.link.tile();
        if change.action.kind != ActionKind::Move && tile == prev.link.tile() {
            return;
        }
        let Some(wf) = c.prev_wavefront else {
            return;
        };
        let (Some(old), Some(new)) = (wf.get(prev.link.tile()), wf.get(tile)) else {
            return;
        };
        let shaped = (old - new) as f64 / PBRS_SCALE;
        if shaped > 0.0 {
            rewards.add(Outcome::reward("reward-pbrs-movement", shaped));
        } else if shaped < 0.0 {
            rewards.add(Outcome::penalty("penalty-pbrs-movement", shaped));
        }
    }
}
