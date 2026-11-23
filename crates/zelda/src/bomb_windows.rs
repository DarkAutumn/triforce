//! The owner's two bomb metrics (Nov 23; docs/architecture.md "Bomb windows"): how often Link takes the bomb
//! chances the game offers, judged by what the observation tells him and nothing the network can't know ("We are
//! testing behavior based on the obs flags we tell the agent, not on magic unknown knowledge to it").
//!
//! - **Metric A, the 10th-kill bomb** ([`tenth_bomb_window`]): the next kill is the 10th damage-free one (`$50` =
//!   9), Link holds a bomb and the family's bomb rule leaves BOMB on. Made with a bomb, that kill arms a forced bomb
//!   drop (asm:Damage_KillMonster sets `$51`). Today's observation shows the streak (`next_kill_arms`,
//!   `help_count`, `drop_is_bombs`).
//! - **Metric B, the bomb carrier** ([`bomb_carrier_window`]): the next death's drop is read from the table (no
//!   forced drop overrides it), the kill cycle points at a bomb column of drop group 2, the bag has room and the
//!   room holds a carrier that can die (a choice: the objective leaves monsters alive, or a non-carrier lives too).
//!   This is the future observation flag "next kill may drop bombs" (backlog): the predicate is written to be the
//!   flag; today it feeds the metric only.
//!
//! **Frame by frame, from RAM alone** (the owner, Nov 23: old versions' replays are scored by the same rules):
//! [`BombWindows`] follows both windows over the play frames of a run ([`frame_kind`]), each pair of consecutive
//! play frames one [`Step`], and counts how each window ended ([`BombWindowStats`]). What RAM does not hold (the
//! family's bomb rule, whether the route's objective fights the room) comes from a [`WindowRules`]: `env` answers it
//! from the route data of the tree that played the run, the same way live in the chain and offline over a replay.
//!
//! The game's rules (zelda-asm `src/bank4/drop-items.asm`, `src/bank7/object-update-table.asm`,
//! `src/bank1/weapon-damage-death.asm`):
//! - `$50` HelpDropCount goes up when the killing hit lands (asm:Damage_KillMonster, to `$0A`); harm
//!   (asm:Link_BeHarmed) zeroes it, `$51` and `$627` WorldKillCount. `$51` is set only by the bomb kill that takes
//!   `$50` to `$0A`.
//! - `$52A` WorldKillCycle goes up (0-9, round) when the death sparkle ends (asm:Obj_EndMetastate), for every type
//!   but the dead dummy `$5D`, a child Gel `$14` and a Red Keese `$1C`; the drop is then decided at once
//!   (asm:Drop_SetUpItem): a fairy if WorldKillCount = `$10`, else the forced drop if `$50` >= `$0A` (bombs if `$51`,
//!   else 5 rupees), else `Drop_ItemTable[group * 10 + cycle]` if the slot's random byte is under the group's rate
//!   (group 2: `$68`, 41%). A Stalfos or Gibdo in slot 1 (the room item's carrier) drops nothing.

use crate::enums::{AnimationKind, AnimationState, Direction, ObjectClass, SwordKind, enemy, item};
use crate::game::{Enemy, GameView};
use std::sync::Arc;

/// Drop group 2 (asm:Drop_Group2Types): blue Octoroks `$09`/`$0A`, Moblin `$03`, Lynel `$01`, Vire `$12`, Goriya
/// `$06`, Darknut `$0B`, Red Wizzrobe `$24`, Gibdo `$30`. Its row has bombs in columns 1, 6 and 8.
pub const BOMB_CARRIER_TYPES: [u8; 9] = [
    enemy::BLUE_SLOW_OCTOROK,
    enemy::BLUE_OCTOROK,
    enemy::BLUE_MOBLIN,
    enemy::BLUE_LYNEL,
    enemy::VIRE,
    enemy::RED_GORIYA,
    enemy::RED_DARKNUT,
    enemy::RED_WIZZROBE,
    enemy::GIBDO,
];

/// WorldKillCycle values whose next death reads a bomb column of group 2 (the cycle moves on first: 0 → column 1,
/// 5 → 6, 7 → 8; asm:Drop_ItemTable).
pub const BOMB_CYCLES: [u8; 3] = [0, 5, 7];

/// Types whose death moves neither the cycle nor the room's kill count (asm:Obj_EndMetastate): the dead dummy, a
/// child Gel, a Red Keese.
const NO_CYCLE_TYPES: [u8; 3] = [0x5D, enemy::GREEN_GEL, enemy::RED_KEESE];

/// ObjType, ObjMetastate (`$10`-`$14` dying) and Item_ObjMonsterType (the type that died in the slot, written when
/// its sparkle ends): per object slot.
const OBJ_TYPE: u16 = 0x34F;
const OBJ_METASTATE: u16 = 0x405;
const ITEM_OBJ_MONSTER_TYPE: u16 = 0x412;
/// A dying monster's slot turns into this type when its sparkle ends (asm:Obj_EndMetastate).
const DROPPED_ITEM: u8 = 0x60;
const MONSTER_SLOTS: std::ops::RangeInclusive<u16> = 1..=0xB;

/// What a frame is to the metrics, by its game mode (`$12`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// Counted: play (5), an underworld cellar (9), a cave (11), Link dying (17), game over (8).
    Play,
    /// A room change under way: a dungeon's entry (2, 3), a scroll (4, 6, 7), stairs (10), a cave's door (16). A
    /// window that needs Link in the room ends with it, even a maze loop's (the same room id scrolled in again).
    Transition,
    /// Neither (title, file select, the triforce held up, the ending, ...): skipped.
    Other,
}

/// The [`FrameKind`] of game mode `mode`.
pub const fn frame_kind(mode: u8) -> FrameKind {
    match mode {
        5 | 8 | 9 | 11 | 17 => FrameKind::Play,
        2 | 3 | 4 | 6 | 7 | 10 | 16 => FrameKind::Transition,
        _ => FrameKind::Other,
    }
}

/// What the metrics need beyond RAM, asked for the state at a step's start, and only when the answer matters.
pub trait WindowRules {
    /// The acting family's bomb rule leaves BOMB on for the 10th kill (`env::config::BombUse::allows` at `$50` = 9:
    /// every mode but `never`).
    fn tenth_rule(&mut self, state: &Arc<GameView>) -> bool;
    /// (BOMB allowed by the family's rule now, the route objective is FIGHT: every monster must die).
    fn carrier_rules(&mut self, state: &Arc<GameView>) -> (bool, bool);
}

/// Fixed answers (tests, a caller without route data).
#[derive(Clone, Copy, Debug)]
pub struct FixedRules {
    pub bomb_rule: bool,
    pub kill_all: bool,
}

impl WindowRules for FixedRules {
    fn tenth_rule(&mut self, _: &Arc<GameView>) -> bool {
        self.bomb_rule
    }

    fn carrier_rules(&mut self, _: &Arc<GameView>) -> (bool, bool) {
        (self.bomb_rule, self.kill_all)
    }
}

/// **Metric A's window** (the 10th-kill bomb): `$50` = 9, Link holds at least one bomb and `bomb_rule`
/// ([`WindowRules::tenth_rule`]). A full bag counts (today's observation lights the streak the same with it;
/// [`BombWindowStats`] counts it apart).
pub fn tenth_bomb_window(state: &GameView, bomb_rule: bool) -> bool {
    bomb_rule && state.next_kill_arms_drop() && state.bombs() >= 1
}

/// A death of `kind` moves WorldKillCycle on.
fn advances_cycle(kind: u8) -> bool {
    !NO_CYCLE_TYPES.contains(&kind)
}

/// A group-2 type outside slot 1's room-item exception: its death reads the group-2 row (whatever weapon).
fn carrier_type(kind: u8, slot: u8) -> bool {
    BOMB_CARRIER_TYPES.contains(&kind) && !(slot == 1 && matches!(kind, enemy::STALFOS | enemy::GIBDO))
}

/// A Vire dies only to a hit of `$40` or more (the magic sword or its beam, a bomb, the silver arrow); any weaker
/// hit splits it into two Red Keese, whose deaths move nothing (zelda-asm enemies.md "Vire"). The metric counts the
/// magic sword and BOMB (allowed by the rule, a bomb in the bag).
fn vire_dies(state: &GameView, bomb_rule: bool) -> bool {
    state.sword() == SwordKind::Magic || (bomb_rule && state.bombs() > 0)
}

/// `e` is an **eligible carrier**: a group-2 type, not the slot-1 room-item Gibdo, and a Vire only where Link can
/// kill it outright ([`vire_dies`]).
pub fn is_bomb_carrier(state: &GameView, e: &Enemy, bomb_rule: bool) -> bool {
    carrier_type(e.id, e.index) && (e.id != enemy::VIRE || vire_dies(state, bomb_rule))
}

/// `e` is the other side of the choice: a monster whose death moves the cycle and reads no bomb column (a
/// non-carrier, the slot-1 Gibdo or Stalfos). A Vire that only splits, a Red Keese and a child Gel are neither; the
/// big Digdogger dies only after the recorder splits it. Hazards (Bubbles, traps, fires, Gleeok's flying heads) and
/// shots are not in `state.enemies` at all.
fn is_other_kill(e: &Enemy) -> bool {
    !carrier_type(e.id, e.index) && e.id != enemy::VIRE && advances_cycle(e.id) && !e.is_big_digdogger()
}

/// What [`bomb_carrier_window`] tests without the rules: the cycle, no death pending, no forced drop, room in the bag,
/// a carrier type.
fn carrier_window_possible(state: &GameView) -> bool {
    BOMB_CYCLES.contains(&state.world_kill_cycle())
        && !death_pending(state)
        && state.help_drop_count() < 9
        && state.kill_streak() != 15
        && state.bombs() < state.bomb_max()
        && state.enemies.iter().any(|e| carrier_type(e.id, e.index))
}

/// **Metric B's window**, the future observation flag "next kill may drop bombs": every one of
/// - WorldKillCycle reads 0, 5 or 7 ([`BOMB_CYCLES`]), and no death that will move it is under way (a monster in its
///   death sparkle: the cycle then speaks for a kill already made, and the flag waits the sparkle's ~19 frames);
/// - no forced drop overrides the table: `$50` < 9 (at 9 with bombs Metric A applies, without them the 10th kill
///   forces 5 rupees; at 10 the forced drop is armed), and WorldKillCount is not 15 (the next kill makes it 16:
///   a fairy);
/// - the bag is not full;
/// - the room holds an eligible carrier ([`is_bomb_carrier`]; `bomb_rule`: BOMB allowed now by the acting family's
///   rule, for a Vire);
/// - a choice exists: `kill_all` (the route objective is FIGHT: every monster must die) is off, or a monster of the
///   other side lives too.
pub fn bomb_carrier_window(state: &GameView, bomb_rule: bool, kill_all: bool) -> bool {
    carrier_window_possible(state)
        && state.enemies.iter().any(|e| is_bomb_carrier(state, e, bomb_rule))
        && (!kill_all || state.enemies.iter().any(is_other_kill))
}

/// [`bomb_carrier_window`] asking `rules` only when an answer can change it (a Vire without the magic sword, or no
/// monster of the other side).
fn carrier_window_with(state: &Arc<GameView>, rules: &mut dyn WindowRules) -> bool {
    if !carrier_window_possible(state) {
        return false;
    }
    let vire = state.sword() != SwordKind::Magic && state.enemies.iter().any(|e| e.id == enemy::VIRE);
    let other = state.enemies.iter().any(is_other_kill);
    let (bomb_rule, kill_all) = if vire || !other { rules.carrier_rules(state) } else { (false, false) };
    bomb_carrier_window(state, bomb_rule, kill_all)
}

/// A bomb wall or bomb secret opened in Link's room between two states: a dungeon bomb door turned open, or the
/// overworld screen's secret flag (`$80` of its room flags) went up.
pub fn opened_wall(prev: &GameView, curr: &GameView) -> bool {
    if prev.full_location() != curr.full_location() || curr.in_cave() {
        return false;
    }
    if curr.level() == 0 {
        let flag = |s: &GameView| {
            s.peek(crate::room_entry::ROOM_FLAGS_OW + u16::from(s.location() & 0x7F)) & 0x80 != 0
        };
        return flag(curr) && !flag(prev);
    }
    [Direction::N, Direction::S, Direction::W, Direction::E]
        .into_iter()
        .any(|d| curr.is_bomb_door_open(d) && !prev.is_bomb_door_open(d))
}

/// Two consecutive play frames (or any two states a step apart): what the windows are judged on.
#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    pub prev: &'a Arc<GameView>,
    pub curr: &'a Arc<GameView>,
    /// Link left the room between them (another room, or a transition frame: a maze loop).
    pub left: bool,
    /// Frames from `prev` to `curr`.
    pub frames: u32,
}

impl Step<'_> {
    /// The 10th damage-free kill (`$50` 9 → 10, or 9 → 0 when the forced drop already fell and the kill count went
    /// up; as `crate::change::Change::tenth_kill`).
    pub fn tenth_kill(&self) -> bool {
        self.prev.next_kill_arms_drop()
            && match self.curr.help_drop_count() {
                10 => true,
                0 => self.curr.kill_streak() > self.prev.kill_streak(),
                _ => false,
            }
    }

    /// The 10th kill was a bomb's: `$51` set, or the forced drop already fell as bombs.
    pub fn armed_bomb_drop(&self) -> bool {
        self.tenth_kill()
            && (self.curr.help_drop_is_bombs()
                || (self.curr.help_drop_count() == 0 && !self.new_bomb_drops().is_empty()))
    }

    /// Bomb slots that went live (only Link places bombs).
    fn placed_bombs(&self) -> impl Iterator<Item = AnimationKind> + '_ {
        [AnimationKind::Bomb1, AnimationKind::Bomb2].into_iter().filter(|&k| {
            self.curr.animation_state(k) != AnimationState::Inactive
                && self.prev.animation_state(k) == AnimationState::Inactive
        })
    }

    /// The bomb drops (slots) that appeared in Link's room.
    fn new_bomb_drops(&self) -> Vec<i8> {
        if self.left {
            return Vec::new();
        }
        let was = |slot: i8| {
            self.prev.items.iter().find(|i| i.index == slot && i.id == item::BOMBS).map(|i| i.timer)
        };
        self.curr
            .items
            .iter()
            .filter(|i| i.id == item::BOMBS && was(i.index).is_none_or(|old| i.timer > old))
            .map(|i| i.index)
            .collect()
    }
}

/// One death that moved the kill cycle within a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Death {
    pub slot: u8,
    /// The type that died.
    pub kind: u8,
    /// The item its slot dropped, when still there at the step's end.
    pub dropped: Option<i16>,
}

fn dying(state: &GameView, slot: u16) -> bool {
    (0x10..=0x14).contains(&state.peek(OBJ_METASTATE + slot))
}

/// How far WorldKillCycle moved within the step, and the deaths that moved it, by slot: a slot that turned into a
/// dropped item (the type from Item_ObjMonsterType), or one whose sparkle ended with no drop (freed); a monster
/// freed without a sparkle seen (killed and gone within a longer step) fills up to the count when
/// Item_ObjMonsterType names it. Leaving the room within the step, the slots that were dying. Fewer deaths than
/// moves: the rest are unattributed.
pub fn cycle_deaths(step: &Step) -> (u8, Vec<Death>) {
    let (p, c) = (step.prev, step.curr);
    let moves = (i32::from(c.world_kill_cycle()) - i32::from(p.world_kill_cycle())).rem_euclid(10) as u8;
    if moves == 0 {
        return (0, Vec::new());
    }
    let (mut deaths, mut maybe) = (Vec::new(), Vec::new());
    for slot in MONSTER_SLOTS {
        let (was, now) = (p.peek(OBJ_TYPE + slot), c.peek(OBJ_TYPE + slot));
        let death = |kind: u8, dropped: Option<i16>| Death { slot: slot as u8, kind, dropped };
        if step.left {
            if dying(p, slot) && advances_cycle(was) {
                deaths.push(death(was, None));
            }
        } else if now == DROPPED_ITEM && was != DROPPED_ITEM {
            let kind = c.peek(ITEM_OBJ_MONSTER_TYPE + slot);
            if advances_cycle(kind) {
                deaths.push(death(kind, c.item_by_index(slot as i8).map(|i| i.id)));
            }
        } else if dying(p, slot) && was != DROPPED_ITEM && (now != was || !dying(c, slot)) {
            if advances_cycle(was) {
                deaths.push(death(was, None));
            }
        } else if now == 0
            && ObjectClass::of(was) == ObjectClass::Monster
            && !matches!(was, enemy::ZORA | enemy::VIRE)
            && advances_cycle(was)
            && c.peek(ITEM_OBJ_MONSTER_TYPE + slot) == was
        {
            maybe.push(death(was, None));
        }
    }
    let room = usize::from(moves).saturating_sub(deaths.len());
    deaths.extend(maybe.into_iter().take(room));
    (moves, deaths)
}

/// A dying monster in `state` whose death will move the cycle.
fn death_pending(state: &GameView) -> bool {
    MONSTER_SLOTS.into_iter().any(|slot| dying(state, slot) && advances_cycle(state.peek(OBJ_TYPE + slot)))
}

/// Dropped bombs followed to their fate (as `crate::bombs::BombStats` follows every bomb drop): taken (gone while
/// the lifetime could not have run out, or Link's count rose) or wasted (expired, or left on the floor).
#[derive(Clone, Debug, Default)]
struct Followed(Vec<(i8, u8)>);

impl Followed {
    fn add(&mut self, state: &GameView, slot: i8) {
        if let Some(i) = state.items.iter().find(|i| i.index == slot && i.id == item::BOMBS) {
            self.0.push((slot, i.timer));
        }
    }

    /// (taken, wasted) within the step.
    fn update(&mut self, step: &Step) -> (u32, u32) {
        if step.left {
            return (0, std::mem::take(&mut self.0).len() as u32);
        }
        let (p, c) = (step.prev, step.curr);
        let reach = step.frames / 2 + crate::hearts::TIMER_MARGIN;
        let gained = c.bombs() > p.bombs();
        let (mut taken, mut wasted) = (0, 0);
        self.0.retain_mut(|(slot, timer)| {
            match c.items.iter().find(|i| i.index == *slot && i.id == item::BOMBS && i.timer <= *timer) {
                Some(i) => {
                    *timer = i.timer;
                    true
                }
                None => {
                    if u32::from(*timer) > reach || gained {
                        taken += 1;
                    } else {
                        wasted += 1;
                    }
                    false
                }
            }
        });
        (taken, wasted)
    }
}

/// Metric A's window over a run: opened at a step that starts inside [`tenth_bomb_window`], paused while Link is
/// on 9 without a bomb in the bag or in the air (or the rule masks BOMB), resumed as the same opportunity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Tenth {
    #[default]
    Closed,
    Open { id: u32, full: bool },
    Paused { id: u32, full: bool },
}

/// A bomb placed while Metric A's window `id` was open, live until its animation slot goes inactive.
#[derive(Clone, Copy, Debug)]
struct LiveBomb {
    kind: AnimationKind,
    id: u32,
    opened_wall: bool,
}

/// How each window of a room visit ended (chain records and replay scores: JSON keys as named; `fields`). Metric A's
/// opportunities are `tenth_armed + tenth_missed + tenth_other`, Metric B's `carrier_hits + carrier_miss_kill +
/// carrier_miss_leave`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BombWindowStats {
    /// The 10th kill made by a bomb: the forced bomb drop armed.
    pub tenth_armed: u32,
    /// A bomb placed in the window that ended (exploded, or was gone with the room) without the 10th kill and
    /// without opening a wall.
    pub tenth_missed: u32,
    /// The 10th kill made by any other weapon.
    pub tenth_other: u32,
    /// The three above, of windows that opened with the bag full.
    pub tenth_full_armed: u32,
    pub tenth_full_missed: u32,
    pub tenth_full_other: u32,
    /// Windows the streak was lost in (harm) before any of the three: not opportunities.
    pub tenth_lost: u32,
    /// Paused windows ended by the 10th kill or harm while Link had no bomb: not opportunities.
    pub tenth_paused_out: u32,
    /// Forced bomb drops that fell, then were taken or wasted (expired, left on the floor); armed drops lost
    /// before falling (harm, or the fairy at 16 kills).
    pub tenth_forced_fell: u32,
    pub tenth_forced_taken: u32,
    pub tenth_forced_wasted: u32,
    pub tenth_forced_lost: u32,
    /// The next death that moved the cycle was an eligible carrier's.
    pub carrier_hits: u32,
    /// It was any other monster's.
    pub carrier_miss_kill: u32,
    /// Link left the room first.
    pub carrier_miss_leave: u32,
    /// The window's conditions ended with no death pending (bag filled, the streak reached 9, the carriers gone
    /// without a death that moved the cycle): not opportunities.
    pub carrier_closed: u32,
    /// Bombs the hits dropped (the 41% roll), and those taken.
    pub carrier_bombs: u32,
    pub carrier_bombs_taken: u32,
    /// Cycle moves in an open window no slot accounted for (expected 0).
    pub carrier_unattributed: u32,
}

impl BombWindowStats {
    pub fn fields(&self) -> [(&'static str, u32); 19] {
        [
            ("tenth_armed", self.tenth_armed),
            ("tenth_missed", self.tenth_missed),
            ("tenth_other", self.tenth_other),
            ("tenth_full_armed", self.tenth_full_armed),
            ("tenth_full_missed", self.tenth_full_missed),
            ("tenth_full_other", self.tenth_full_other),
            ("tenth_lost", self.tenth_lost),
            ("tenth_paused_out", self.tenth_paused_out),
            ("tenth_forced_fell", self.tenth_forced_fell),
            ("tenth_forced_taken", self.tenth_forced_taken),
            ("tenth_forced_wasted", self.tenth_forced_wasted),
            ("tenth_forced_lost", self.tenth_forced_lost),
            ("carrier_hits", self.carrier_hits),
            ("carrier_miss_kill", self.carrier_miss_kill),
            ("carrier_miss_leave", self.carrier_miss_leave),
            ("carrier_closed", self.carrier_closed),
            ("carrier_bombs", self.carrier_bombs),
            ("carrier_bombs_taken", self.carrier_bombs_taken),
            ("carrier_unattributed", self.carrier_unattributed),
        ]
    }
}

/// Both windows over one run: [`BombWindows::frame`] on every counted frame ([`FrameKind::Play`]),
/// [`BombWindows::skip`] on the others; a fresh one per run.
#[derive(Clone, Debug, Default)]
pub struct BombWindows {
    tenth: Tenth,
    opened: u32,
    bombs: Vec<LiveBomb>,
    carrier: bool,
    forced: Followed,
    carried: Followed,
    /// A forced bomb drop armed by a 10th kill and not yet fallen (Dodongo's death arms one too, zelda-asm
    /// `src/bank4/dodongo.asm`: not the metric's).
    armed_pending: bool,
    prev: Option<Arc<GameView>>,
    /// A transition frame since `prev`, and frames since it.
    left: bool,
    gap: u32,
    /// Link died: nothing counts after.
    over: bool,
}

impl BombWindows {
    /// (Metric A's window unresolved, open or paused; Metric B's open): at the run's end, `open_at_end`.
    pub fn open(&self) -> (bool, bool) {
        (self.tenth != Tenth::Closed, self.carrier)
    }

    /// The last counted frame's state.
    pub fn last(&self) -> Option<&Arc<GameView>> {
        self.prev.as_ref()
    }

    /// A frame not counted: a transition (`left`) ends Link's stay in the room.
    pub fn skip(&mut self, left: bool) {
        self.gap += 1;
        self.left |= left;
    }

    /// Counts the step from the last counted frame to `state` into `stats` (the room the step started in);
    /// `rules` answer for that step's start.
    pub fn frame(&mut self, state: Arc<GameView>, rules: &mut dyn WindowRules, stats: &mut BombWindowStats) {
        if self.over {
            return;
        }
        if let Some(prev) = self.prev.take() {
            let left = self.left || prev.full_location() != state.full_location();
            self.update(&Step { prev: &prev, curr: &state, left, frames: self.gap + 1 }, rules, stats);
        }
        (self.left, self.gap) = (false, 0);
        self.over = state.game_over();
        self.prev = Some(state);
    }

    /// Counts one step into `stats`.
    pub fn update(&mut self, step: &Step, rules: &mut dyn WindowRules, stats: &mut BombWindowStats) {
        self.update_tenth(step, rules, stats);
        self.update_carrier(step, rules, stats);
    }

    fn update_tenth(&mut self, step: &Step, rules: &mut dyn WindowRules, stats: &mut BombWindowStats) {
        let (p, c) = (step.prev, step.curr);
        let window = p.next_kill_arms_drop() && p.bombs() >= 1 && rules.tenth_rule(p);
        self.tenth = match self.tenth {
            Tenth::Closed if window => {
                self.opened += 1;
                Tenth::Open { id: self.opened, full: p.bombs() >= p.bomb_max() }
            }
            Tenth::Paused { id, full } if window => Tenth::Open { id, full },
            // Still on 9 (harm is closed within its own step), a bomb of the window in the air: still open.
            Tenth::Open { id, full } if !window && p.next_kill_arms_drop() => {
                if self.bombs.iter().any(|b| b.id == id) { Tenth::Open { id, full } } else { Tenth::Paused { id, full } }
            }
            Tenth::Open { .. } if !window => {
                stats.tenth_lost += 1;
                Tenth::Closed
            }
            t => t,
        };
        if let Tenth::Open { id, .. } = self.tenth {
            let placed: Vec<AnimationKind> = step.placed_bombs().collect();
            self.bombs.extend(placed.into_iter().map(|kind| LiveBomb { kind, id, opened_wall: false }));
        }
        let opened = opened_wall(p, c);
        for b in &mut self.bombs {
            b.opened_wall |= opened;
        }
        if step.tenth_kill() {
            match self.tenth {
                Tenth::Open { full, .. } if step.armed_bomb_drop() => {
                    stats.tenth_armed += 1;
                    stats.tenth_full_armed += u32::from(full);
                }
                Tenth::Open { full, .. } => {
                    stats.tenth_other += 1;
                    stats.tenth_full_other += u32::from(full);
                }
                Tenth::Paused { .. } => stats.tenth_paused_out += 1,
                Tenth::Closed => {}
            }
            self.tenth = Tenth::Closed;
            self.bombs.clear();
        } else if self.tenth != Tenth::Closed && !c.next_kill_arms_drop() && !c.game_over() {
            match self.tenth {
                Tenth::Paused { .. } => stats.tenth_paused_out += 1,
                _ => stats.tenth_lost += 1,
            }
            self.tenth = Tenth::Closed;
            self.bombs.clear();
        }
        let current = match self.tenth {
            Tenth::Open { id, full } | Tenth::Paused { id, full } => Some((id, full)),
            Tenth::Closed => None,
        };
        let mut missed = None;
        self.bombs.retain(|b| {
            let ended = step.left || c.animation_state(b.kind) == AnimationState::Inactive;
            if ended && !b.opened_wall && current.is_some_and(|(id, _)| id == b.id) {
                missed = current.map(|(_, full)| full);
            }
            !ended
        });
        if let Some(full) = missed {
            stats.tenth_missed += 1;
            stats.tenth_full_missed += u32::from(full);
            self.tenth = Tenth::Closed;
            self.bombs.clear();
        }
        let (taken, wasted) = self.forced.update(step);
        stats.tenth_forced_taken += taken;
        stats.tenth_forced_wasted += wasted;
        let armed = |s: &GameView| s.help_drop_count() >= 10 && s.help_drop_is_bombs();
        self.armed_pending |= step.armed_bomb_drop();
        if self.armed_pending && !armed(c) {
            self.armed_pending = false;
            let drops = step.new_bomb_drops();
            if drops.is_empty() {
                stats.tenth_forced_lost += 1;
            } else {
                stats.tenth_forced_fell += 1;
                for slot in drops {
                    self.forced.add(c, slot);
                }
            }
        }
    }

    fn update_carrier(&mut self, step: &Step, rules: &mut dyn WindowRules, stats: &mut BombWindowStats) {
        let (p, c) = (step.prev, step.curr);
        let window = carrier_window_with(p, rules);
        if self.carrier && !window && !death_pending(p) {
            stats.carrier_closed += 1;
            self.carrier = false;
        }
        self.carrier |= window;
        let (taken, _) = self.carried.update(step);
        stats.carrier_bombs_taken += taken;
        let (moves, deaths) = cycle_deaths(step);
        if moves > 0 {
            if self.carrier {
                stats.carrier_unattributed += u32::from(moves).saturating_sub(deaths.len() as u32);
                let hits: Vec<&Death> = deaths.iter().filter(|d| carrier_type(d.kind, d.slot)).collect();
                if hits.is_empty() {
                    stats.carrier_miss_kill += 1;
                } else {
                    stats.carrier_hits += 1;
                    for d in hits.into_iter().filter(|d| d.dropped == Some(item::BOMBS)) {
                        stats.carrier_bombs += 1;
                        self.carried.add(c, d.slot as i8);
                    }
                }
                self.carrier = false;
            }
        } else if self.carrier && step.left {
            stats.carrier_miss_leave += 1;
            self.carrier = false;
        }
    }
}

#[cfg(test)]
mod tests {
    //! Hand-built state sequences, one per resolution rule of the owner's spec (no emulator).
    use super::*;
    use crate::executor::tests::playable;
    use crate::game_map::GameMap;
    use crate::tables::{self, RAM_SIZE};
    use crate::vars::Var;

    type Ram = [u8; 0x800];

    fn view(ram: &Ram, frame: u64) -> Arc<GameView> {
        static MAP: std::sync::LazyLock<Arc<GameMap>> = std::sync::LazyLock::new(|| Arc::new(GameMap::builtin()));
        let mut image = [0u8; RAM_SIZE];
        image[..0x800].copy_from_slice(ram);
        Arc::new(GameView::new(&image, frame, false, MAP.clone()))
    }

    fn set(ram: &mut Ram, var: Var, value: u8) {
        ram[var.addr() as usize] = value;
    }

    /// Playable in 1:0x44 with `bombs` of 8, `help` damage-free kills, the cycle at `cycle`.
    fn base(bombs: u8, help: u8, cycle: u8) -> Ram {
        let mut ram = playable();
        set(&mut ram, Var::Bombs, bombs);
        set(&mut ram, Var::BombMax, 8);
        set(&mut ram, Var::HelpDropCount, help);
        set(&mut ram, Var::KillStreak, help);
        set(&mut ram, Var::WorldKillCycle, cycle);
        ram
    }

    fn monster(ram: &mut Ram, slot: usize, id: u8) {
        ram[tables::OBJ_ID.0 + slot] = id;
        ram[tables::OBJ_HEALTH.0 + slot] = 0x20;
        (ram[tables::OBJ_POS_X.0 + slot], ram[tables::OBJ_POS_Y.0 + slot]) = (40, 100 + slot as u8 * 8);
    }

    /// `slot`'s monster died within the step: its sparkle ended (the cycle moved on) and the slot holds `drop` (or
    /// was freed).
    fn died(ram: &mut Ram, slot: usize, drop: Option<i16>) {
        let kind = ram[tables::OBJ_ID.0 + slot];
        ram[ITEM_OBJ_MONSTER_TYPE as usize + slot] = kind;
        ram[tables::OBJ_ID.0 + slot] = if drop.is_some() { DROPPED_ITEM } else { 0 };
        ram[tables::OBJ_STATUS.0 + slot] = drop.unwrap_or(0) as u8;
        ram[tables::ITEM_TIMER.0 + slot] = 0xFF;
        ram[Var::WorldKillCycle.addr() as usize] = (ram[Var::WorldKillCycle.addr() as usize] + 1) % 10;
    }

    fn kill(ram: &mut Ram, slot: usize, drop: Option<i16>) {
        died(ram, slot, drop);
        ram[Var::HelpDropCount.addr() as usize] += 1;
        ram[Var::KillStreak.addr() as usize] += 1;
    }

    /// Feeds `rams` as consecutive play frames.
    fn play(w: &mut BombWindows, rams: &[Ram], bomb_rule: bool, kill_all: bool) -> BombWindowStats {
        let mut stats = BombWindowStats::default();
        let mut rules = FixedRules { bomb_rule, kill_all };
        for (i, ram) in rams.iter().enumerate() {
            w.frame(view(ram, 100 + i as u64), &mut rules, &mut stats);
        }
        stats
    }

    const BOMB_ACTIVE: u8 = 18;
    const BOMB_EXPLODED: u8 = 19;

    /// A bomb placed from `ram` (one fewer in the bag, slot 1 active).
    fn placed(ram: &Ram) -> Ram {
        let mut next = *ram;
        set(&mut next, Var::Bombs, ram[Var::Bombs.addr() as usize] - 1);
        set(&mut next, Var::BombOrFlameAnimation, BOMB_ACTIVE);
        next
    }

    fn bomb(ram: &Ram, state: u8) -> Ram {
        let mut next = *ram;
        set(&mut next, Var::BombOrFlameAnimation, state);
        next
    }

    #[test]
    fn tenth_window_needs_nine_a_bomb_and_the_rule() {
        assert!(tenth_bomb_window(&view(&base(1, 9, 3), 0), true));
        assert!(!tenth_bomb_window(&view(&base(1, 9, 3), 0), false), "bombs: never");
        assert!(!tenth_bomb_window(&view(&base(0, 9, 3), 0), true), "no bomb");
        assert!(!tenth_bomb_window(&view(&base(4, 8, 3), 0), true), "not the 10th");
    }

    /// A: the 10th kill by a bomb arms the drop; the forced drop then falls and is taken.
    #[test]
    fn bomb_kill_arms() {
        let mut start = base(3, 9, 3);
        monster(&mut start, 2, enemy::STALFOS);
        let s1 = placed(&start);
        let s2 = bomb(&s1, BOMB_EXPLODED);
        let mut s3 = bomb(&s2, 0);
        s3[tables::OBJ_SPAWN_STATE.0 + 2] = 0x10;
        set(&mut s3, Var::HelpDropCount, 10);
        set(&mut s3, Var::HelpDropValue, 1);
        let mut s4 = s3;
        s4[tables::OBJ_SPAWN_STATE.0 + 2] = 0;
        died(&mut s4, 2, Some(item::BOMBS));
        set(&mut s4, Var::HelpDropCount, 0);
        set(&mut s4, Var::HelpDropValue, 0);
        let mut s5 = s4;
        s5[tables::OBJ_ID.0 + 2] = 0;
        set(&mut s5, Var::Bombs, 6);
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, s1, s2, s3, s4, s5], true, false);
        assert_eq!((stats.tenth_armed, stats.tenth_missed, stats.tenth_other, stats.tenth_lost), (1, 0, 0, 0));
        assert_eq!((stats.tenth_forced_fell, stats.tenth_forced_taken, stats.tenth_forced_lost), (1, 1, 0));
        assert_eq!(w.open().0, false);
    }

    /// A: a bomb that explodes killing nothing is a miss; Link still on 9 with bombs: a new opportunity, which the
    /// sword then resolves (other).
    #[test]
    fn missed_bomb_then_a_new_chance() {
        let mut start = base(3, 9, 3);
        monster(&mut start, 2, enemy::STALFOS);
        let s1 = placed(&start);
        let s2 = bomb(&s1, BOMB_EXPLODED);
        let s3 = bomb(&s2, 0);
        let mut s4 = s3;
        kill(&mut s4, 2, None);
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, s1, s2, s3, s4], true, false);
        assert_eq!((stats.tenth_armed, stats.tenth_missed, stats.tenth_other), (0, 1, 1));
    }

    /// A: a bomb that only opens a wall is ignored (the window stays open); the same window then ends by a bomb kill.
    /// An overworld secret (room flag `$80`) is the wall here.
    #[test]
    fn wall_only_bomb_is_ignored() {
        let mut start = base(3, 9, 3);
        set(&mut start, Var::Level, 0);
        set(&mut start, Var::Location, 0x2c);
        monster(&mut start, 2, enemy::RED_OCTOROK);
        let s1 = placed(&start);
        let mut s2 = bomb(&s1, BOMB_EXPLODED);
        s2[crate::room_entry::ROOM_FLAGS_OW as usize + 0x2c] |= 0x80;
        let s3 = bomb(&s2, 0);
        let s4 = placed(&s3);
        let mut s5 = bomb(&s4, 0);
        kill(&mut s5, 2, None);
        set(&mut s5, Var::HelpDropValue, 1);
        let mut w = BombWindows::default();
        let stats =
            play(&mut w, &[start, s1, s2, s3, s4, s5], true, false);
        assert_eq!((stats.tenth_armed, stats.tenth_missed, stats.tenth_other), (1, 0, 0));
    }

    /// A: the last bomb spent on a wall pauses the window; a 10th kill while paused counts nothing; bombs back on 9:
    /// the same opportunity resumes and the sword's kill is its one resolution.
    #[test]
    fn last_bomb_on_a_wall_pauses() {
        let mut start = base(1, 9, 3);
        set(&mut start, Var::Level, 0);
        set(&mut start, Var::Location, 0x2c);
        monster(&mut start, 2, enemy::RED_OCTOROK);
        let s1 = placed(&start);
        let mut s2 = bomb(&s1, 0);
        s2[crate::room_entry::ROOM_FLAGS_OW as usize + 0x2c] |= 0x80;
        let s3 = s2;
        let mut s4 = s3;
        set(&mut s4, Var::Bombs, 1);
        let s5 = s4;
        let mut s6 = s5;
        kill(&mut s6, 2, None);
        let mut w = BombWindows::default();
        let stats =
            play(&mut w, &[start, s1, s2, s3, s4, s5, s6], true, false);
        assert_eq!((stats.tenth_armed, stats.tenth_missed, stats.tenth_other, stats.tenth_paused_out), (0, 0, 1, 0));
        // Without the bombs back, the sword's 10th kill ends the paused window uncounted.
        let mut s4 = s3;
        kill(&mut s4, 2, None);
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, s1, s2, s3, s4], true, false);
        assert_eq!((stats.tenth_other, stats.tenth_paused_out), (0, 1));
    }

    /// A: harm before any event loses the streak: no opportunity; a full bag is counted apart.
    #[test]
    fn streak_lost_and_full_bag() {
        let mut start = base(8, 9, 3);
        monster(&mut start, 2, enemy::STALFOS);
        let mut hurt = start;
        set(&mut hurt, Var::HelpDropCount, 0);
        set(&mut hurt, Var::KillStreak, 0);
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, start, hurt], true, false);
        assert_eq!((stats.tenth_lost, stats.tenth_armed + stats.tenth_missed + stats.tenth_other), (1, 0));
        let mut killed = start;
        kill(&mut killed, 2, None);
        let stats = play(&mut BombWindows::default(), &[start, killed], true, false);
        assert_eq!((stats.tenth_other, stats.tenth_full_other), (1, 1));
    }

    /// A: the 10th kill by another weapon is the miss; under `bombs: never` there is no window.
    #[test]
    fn other_weapon_tenth_kill() {
        let mut start = base(2, 9, 3);
        monster(&mut start, 2, enemy::STALFOS);
        let mut killed = start;
        kill(&mut killed, 2, None);
        let stats = play(&mut BombWindows::default(), &[start, killed], true, false);
        assert_eq!((stats.tenth_other, stats.tenth_full_other), (1, 0));
        let stats = play(&mut BombWindows::default(), &[start, killed], false, false);
        assert_eq!(stats.tenth_other, 0);
    }

    /// B: the next death that moves the cycle is a carrier's: a hit, its bomb drop counted.
    #[test]
    fn carrier_hit() {
        let mut start = base(2, 3, 5);
        monster(&mut start, 2, enemy::RED_DARKNUT);
        monster(&mut start, 3, enemy::STALFOS);
        assert!(bomb_carrier_window(&view(&start, 0), true, false));
        let mut killed = start;
        kill(&mut killed, 2, Some(item::BOMBS));
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, killed], true, false);
        assert_eq!((stats.carrier_hits, stats.carrier_miss_kill, stats.carrier_bombs), (1, 0, 1));
        assert!(!w.open().1);
        // The cycle now reads 6: no bomb column next.
        assert!(!bomb_carrier_window(&view(&killed, 0), true, false));
    }

    /// B: a non-carrier dies first: a miss by kill. The carrier's sparkle running over the 9th kill keeps the
    /// window open until that death.
    #[test]
    fn non_carrier_miss_and_pending_death() {
        let mut start = base(2, 3, 0);
        monster(&mut start, 2, enemy::RED_DARKNUT);
        monster(&mut start, 3, enemy::STALFOS);
        let mut killed = start;
        kill(&mut killed, 3, None);
        let stats = play(&mut BombWindows::default(), &[start, killed], true, false);
        assert_eq!((stats.carrier_hits, stats.carrier_miss_kill), (0, 1));

        let mut start = base(2, 8, 7);
        monster(&mut start, 2, enemy::RED_DARKNUT);
        let mut hit = start;
        hit[tables::OBJ_SPAWN_STATE.0 + 2] = 0x10;
        set(&mut hit, Var::HelpDropCount, 9);
        let mut gone = hit;
        gone[tables::OBJ_SPAWN_STATE.0 + 2] = 0;
        died(&mut gone, 2, None);
        let stats = play(&mut BombWindows::default(), &[start, hit, gone], true, false);
        assert_eq!((stats.carrier_hits, stats.carrier_closed), (1, 0));
    }

    /// B: leaving the room with the window open is a miss.
    #[test]
    fn leaving_miss() {
        let mut start = base(2, 0, 0);
        monster(&mut start, 2, enemy::BLUE_MOBLIN);
        let mut next = playable();
        set(&mut next, Var::Location, 0x45);
        set(&mut next, Var::Bombs, 2);
        set(&mut next, Var::BombMax, 8);
        let stats = play(&mut BombWindows::default(), &[start, next], true, false);
        assert_eq!((stats.carrier_miss_leave, stats.carrier_hits + stats.carrier_miss_kill), (1, 0));
    }

    /// B: a room that must be cleared with carriers only offers no choice; a non-carrier alive gives one back. A
    /// full bag, the 10th kill's streak, the fairy's and a non-bomb cycle shut it.
    #[test]
    fn kill_all_all_carrier_exclusion_and_forced_drops() {
        let mut darknuts = base(2, 0, 0);
        monster(&mut darknuts, 2, enemy::RED_DARKNUT);
        monster(&mut darknuts, 3, enemy::RED_DARKNUT);
        assert!(!bomb_carrier_window(&view(&darknuts, 0), true, true), "FIGHT, Darknuts only");
        assert!(bomb_carrier_window(&view(&darknuts, 0), true, false), "pass-through");
        let mut mixed = darknuts;
        monster(&mut mixed, 4, enemy::STALFOS);
        assert!(bomb_carrier_window(&view(&mixed, 0), true, true));
        let shut = |f: &dyn Fn(&mut Ram)| {
            let mut ram = mixed;
            f(&mut ram);
            !bomb_carrier_window(&view(&ram, 0), true, false)
        };
        assert!(shut(&|r| set(r, Var::Bombs, 8)), "full bag");
        assert!(shut(&|r| set(r, Var::HelpDropCount, 9)), "Metric A's streak");
        assert!(
            shut(&|r| {
                set(r, Var::Bombs, 0);
                set(r, Var::HelpDropCount, 9);
            }),
            "9 without bombs: rupees"
        );
        assert!(shut(&|r| set(r, Var::KillStreak, 15)), "a fairy next");
        assert!(shut(&|r| set(r, Var::WorldKillCycle, 1)), "column 2");
        // The slot-1 Gibdo carries the room item and drops nothing: no carrier.
        let mut gibdo = base(2, 0, 0);
        monster(&mut gibdo, 1, enemy::GIBDO);
        assert!(!bomb_carrier_window(&view(&gibdo, 0), true, false));
        monster(&mut gibdo, 2, enemy::GIBDO);
        assert!(bomb_carrier_window(&view(&gibdo, 0), true, true), "slot 2's Gibdo, the slot-1 one the choice");
    }

    /// B: Bubbles are no monsters: Red Wizzrobes with Bubbles in a FIGHT room never count.
    #[test]
    fn bubbles_are_ignored() {
        let mut ram = base(2, 0, 0);
        monster(&mut ram, 2, enemy::RED_WIZZROBE);
        monster(&mut ram, 3, enemy::BUBBLE);
        monster(&mut ram, 4, enemy::BLUE_BUBBLE);
        assert!(!bomb_carrier_window(&view(&ram, 0), true, true));
        assert!(bomb_carrier_window(&view(&ram, 0), true, false));
    }

    /// B: a Vire is a carrier only with the magic sword or BOMB allowed with a bomb in the bag (else it splits).
    #[test]
    fn vire_needs_the_magic_sword_or_bombs() {
        let mut ram = base(0, 0, 0);
        monster(&mut ram, 2, enemy::VIRE);
        set(&mut ram, Var::Sword, 1);
        assert!(!bomb_carrier_window(&view(&ram, 0), true, false), "wood sword, no bombs");
        set(&mut ram, Var::Bombs, 2);
        assert!(!bomb_carrier_window(&view(&ram, 0), false, false), "bombs masked");
        assert!(bomb_carrier_window(&view(&ram, 0), true, false), "bombs allowed");
        set(&mut ram, Var::Bombs, 0);
        set(&mut ram, Var::Sword, 3);
        assert!(bomb_carrier_window(&view(&ram, 0), false, false), "magic sword");
        // A splitting Vire is no choice either: Vire and Stalfos under FIGHT with the wood sword and no bombs.
        set(&mut ram, Var::Sword, 1);
        monster(&mut ram, 3, enemy::STALFOS);
        assert!(!bomb_carrier_window(&view(&ram, 0), true, true));
    }

    /// B: a transition frame between two play frames of the same room (a maze loop scrolls the room in again) is a
    /// leaving; an item sub-screen or any other skipped frame is not.
    #[test]
    fn a_transition_frame_is_a_leaving() {
        assert_eq!(frame_kind(7), FrameKind::Transition);
        assert_eq!(frame_kind(5), FrameKind::Play);
        assert_eq!(frame_kind(0x12), FrameKind::Other);
        let mut ram = base(2, 0, 0);
        set(&mut ram, Var::Level, 0);
        monster(&mut ram, 2, enemy::BLUE_MOBLIN);
        let mut rules = FixedRules { bomb_rule: true, kill_all: false };
        let mut stats = BombWindowStats::default();
        let mut w = BombWindows::default();
        w.frame(view(&ram, 0), &mut rules, &mut stats);
        w.skip(false);
        w.frame(view(&ram, 2), &mut rules, &mut stats);
        assert_eq!((stats.carrier_miss_leave, w.open().1), (0, true));
        w.skip(true);
        w.frame(view(&ram, 4), &mut rules, &mut stats);
        assert_eq!(stats.carrier_miss_leave, 1);
    }

    /// B: while a death sparkle runs, the cycle speaks for a kill already made: no window opens, and that death
    /// counts for nothing.
    #[test]
    fn no_window_over_a_pending_death() {
        let mut start = base(2, 3, 0);
        monster(&mut start, 2, enemy::RED_DARKNUT);
        monster(&mut start, 3, enemy::STALFOS);
        start[tables::OBJ_SPAWN_STATE.0 + 3] = 0x10;
        assert!(!bomb_carrier_window(&view(&start, 0), true, false));
        let mut gone = start;
        gone[tables::OBJ_SPAWN_STATE.0 + 3] = 0;
        died(&mut gone, 3, None);
        let stats = play(&mut BombWindows::default(), &[start, gone], true, false);
        assert_eq!((stats.carrier_hits, stats.carrier_miss_kill), (0, 0));
    }
}
