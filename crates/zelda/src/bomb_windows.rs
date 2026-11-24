//! The owner's two bomb metrics (Nov 23; docs/architecture.md "Bomb windows"): how often Link takes the bomb
//! chances the game offers, judged by what the observation tells him and nothing the network can't know ("We are
//! testing behavior based on the obs flags we tell the agent, not on magic unknown knowledge to it").
//!
//! - **Metric A, the 10th-kill bomb** ([`tenth_bomb_window`]): the next kill is the 10th damage-free one (`$50` =
//!   9), Link holds a bomb of his own and the bomb mode is not `never`. Made with a bomb, that kill arms a forced
//!   bomb drop (asm:Damage_KillMonster sets `$51`). Today's observation shows the streak (`next_kill_arms`,
//!   `help_count`, `drop_is_bombs`).
//! - **Metric B, the bomb carrier** ([`bomb_carrier_window`]): the next death's drop is read from the table (no
//!   forced drop overrides it), the kill cycle points at a bomb column of drop group 2, the bag has room and the
//!   room holds a carrier that can die (a choice: the objective leaves monsters alive, or a non-carrier lives too).
//!   This is the future observation flag "next kill may drop bombs" (backlog): the predicate is written to be the
//!   flag; today it feeds the metric only.
//!
//! **A calculation over a replay, nothing more** (the owner, Nov 23): [`BombWindows`] takes a replay's frames one by
//! one ([`frame_kind`]: play frames counted, each pair of consecutive ones a [`Step`]; transition frames end Link's
//! stay in a room) and lists every window it saw as a [`Window`] record (metric, open and close frame and room,
//! outcome, what was killed, the bag, the drop); counts are sums over that list ([`BombWindowStats`], and
//! `triforce/bomb_metrics.py`). What RAM does not hold comes from a [`WindowRules`]: the facts the replay's targets
//! stream recorded at each decision (the objective FIGHT, the bomb rule; `env::bomb_windows::StepFacts`), and the
//! cheated bombs come from its EDITS (`env::bomb_windows::BombFrames`). Nothing counts during play.
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

use crate::enums::{AnimationKind, AnimationState, Direction, MapLocation, ObjectClass, SwordKind, enemy, item};
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

/// **Metric A's window** (the 10th-kill bomb): `$50` = 9, Link holds at least one bomb of his own (`own_bombs`: the
/// bag less the cheated wall bombs not yet spent, [`own_bombs`]) and `bomb_rule` ([`WindowRules::tenth_rule`]). A full
/// bag counts (today's observation lights the streak the same with it; each [`Window`] says whether the bag was
/// full when it closed).
pub fn tenth_bomb_window(state: &GameView, own_bombs: i32, bomb_rule: bool) -> bool {
    bomb_rule && state.next_kill_arms_drop() && own_bombs >= 1
}

/// Link's own bombs in `state`: the bag (`$658`) less `granted`, the cheated bombs (`zelda::cheats`, a replay's
/// EDITS raising `$658`) not yet spent or taken back.
pub fn own_bombs(state: &GameView, granted: i32) -> i32 {
    (state.bombs() - granted).max(0)
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
/// magic sword and BOMB (allowed by the rule, a bomb of Link's own in the bag).
fn vire_dies(state: &GameView, own_bombs: i32, bomb_rule: bool) -> bool {
    state.sword() == SwordKind::Magic || (bomb_rule && own_bombs > 0)
}

/// `e` is an **eligible carrier**: a group-2 type, not the slot-1 room-item Gibdo, and a Vire only where Link can
/// kill it outright ([`vire_dies`]).
pub fn is_bomb_carrier(state: &GameView, e: &Enemy, own_bombs: i32, bomb_rule: bool) -> bool {
    carrier_type(e.id, e.index) && (e.id != enemy::VIRE || vire_dies(state, own_bombs, bomb_rule))
}

/// `e` is the other side of the choice: a monster whose death moves the cycle and reads no bomb column (a
/// non-carrier, the slot-1 Gibdo or Stalfos). A Vire that only splits, a Red Keese and a child Gel are neither; the
/// big Digdogger dies only after the recorder splits it. Hazards (Bubbles, traps, fires, Gleeok's flying heads) and
/// shots are not in `state.enemies` at all.
fn is_other_kill(e: &Enemy) -> bool {
    !carrier_type(e.id, e.index) && e.id != enemy::VIRE && advances_cycle(e.id) && !e.is_big_digdogger()
}

/// What [`bomb_carrier_window`] tests without the rules: the cycle, no death pending (so a kill streak of 16 with a
/// death under way, a forced fairy, is out too), no forced drop, room in the bag, a carrier type.
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
/// - the room holds an eligible carrier ([`is_bomb_carrier`]; `own_bombs` and `bomb_rule`, BOMB allowed now by the
///   acting family's rule, for a Vire);
/// - a choice exists: `kill_all` (the objective Link was given is FIGHT: every monster must die) is off, or a monster
///   of the other side lives too.
pub fn bomb_carrier_window(state: &GameView, own_bombs: i32, bomb_rule: bool, kill_all: bool) -> bool {
    carrier_window_possible(state)
        && state.enemies.iter().any(|e| is_bomb_carrier(state, e, own_bombs, bomb_rule))
        && (!kill_all || state.enemies.iter().any(is_other_kill))
}

/// [`bomb_carrier_window`] asking `rules` only when an answer can change it (a Vire without the magic sword, or no
/// monster of the other side).
fn carrier_window_with(state: &Arc<GameView>, own_bombs: i32, rules: &mut dyn WindowRules) -> bool {
    if !carrier_window_possible(state) {
        return false;
    }
    let vire = state.sword() != SwordKind::Magic && state.enemies.iter().any(|e| e.id == enemy::VIRE);
    let other = state.enemies.iter().any(is_other_kill);
    let (bomb_rule, kill_all) = if vire || !other { rules.carrier_rules(state) } else { (false, false) };
    bomb_carrier_window(state, own_bombs, bomb_rule, kill_all)
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
    /// Link's own bombs in `prev` ([`own_bombs`]).
    pub own_prev: i32,
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

/// A monster in `state`'s slot in its death sparkle (ObjMetastate `$10`-`$14`). A slot that holds no monster can keep
/// a dying metastate from an earlier object (the 2.0.0.13 250 test: 248 slots of type `$00` read `$10`-`$14`): that
/// is no death under way.
fn dying(state: &GameView, slot: u16) -> bool {
    (0x10..=0x14).contains(&state.peek(OBJ_METASTATE + slot))
        && ObjectClass::of(state.peek(OBJ_TYPE + slot)) == ObjectClass::Monster
}

/// How far WorldKillCycle moved within the step, and the deaths that moved it, by slot: a slot that turned into a
/// dropped item (the type the slot held in `prev` while dying; Item_ObjMonsterType only when the whole death fell
/// within the step), or one whose sparkle ended with no drop (freed); a monster freed without a sparkle seen (killed
/// and gone within a longer step) fills up to the count when Item_ObjMonsterType names it. Leaving the room within
/// the step, the slots that were dying. Fewer deaths than moves: the rest are unattributed.
///
/// Item_ObjMonsterType (`$412`) is not to be trusted on the frame the slot turns into the item: when the drop is a
/// fairy, asm:Drop_SetUpItem goes on to asm:Fairy_Init in the same call, whose asm:Flyer_ResetState zeroes
/// Flyer_ObjPosFrac, the same byte (a kill streak of 16, or a fairy column of the table: 3:0x5b's Darknut at cycle
/// 9 → 0, 9:0x16's Patra at 4 → 5 in the 2.0.0.13 single run).
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
            let kind = if dying(p, slot) { was } else { c.peek(ITEM_OBJ_MONSTER_TYPE + slot) };
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

/// The monsters a step killed, (type, slot): those that started dying, else those whose slot changed type (killed
/// and gone within a longer step). None when Link left the room.
fn new_kills(step: &Step) -> Vec<(u8, u8)> {
    if step.left {
        return Vec::new();
    }
    let (p, c) = (step.prev, step.curr);
    let alive = |slot: u16| ObjectClass::of(p.peek(OBJ_TYPE + slot)) == ObjectClass::Monster && !dying(p, slot);
    let kill = |slot: u16| (p.peek(OBJ_TYPE + slot), slot as u8);
    let started: Vec<_> = MONSTER_SLOTS.filter(|&s| alive(s) && dying(c, s)).map(kill).collect();
    if !started.is_empty() {
        return started;
    }
    MONSTER_SLOTS.filter(|&s| alive(s) && c.peek(OBJ_TYPE + s) != p.peek(OBJ_TYPE + s)).map(kill).collect()
}

/// Dodongo is in the room (types `$31`, `$32`): his room is outside both metrics (his own bomb rules).
fn dodongo_here(state: &GameView) -> bool {
    MONSTER_SLOTS.into_iter().any(|s| matches!(state.peek(OBJ_TYPE + s), enemy::DODONGO | enemy::DODONGO2))
}

/// The bag is full (`$658` ≥ `$67C`).
fn bag_full(state: &GameView) -> bool {
    state.bombs() >= state.bomb_max()
}

/// Which metric a [`Window`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// A, the 10th-kill bomb.
    Tenth,
    /// B, the bomb carrier.
    Carrier,
}

impl Metric {
    pub const fn name(self) -> &'static str {
        match self {
            Metric::Tenth => "tenth",
            Metric::Carrier => "carrier",
        }
    }
}

/// How a [`Window`] ended. [`Outcome::opportunity`]: the six that count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A: the 10th kill by a bomb (the forced bomb drop armed).
    Armed,
    /// A: a bomb placed in the window ended (exploded, or was gone when Link left) with no 10th kill and no wall
    /// opened.
    Missed,
    /// A: the 10th kill by any other weapon, whatever it killed (`killed` names it).
    Other,
    /// A: harm lost the streak first.
    Lost,
    /// A: a paused window (no bomb of Link's own, or the rule masks BOMB) ended by the 10th kill or harm.
    PausedOut,
    /// B: the next death that moved the cycle was an eligible carrier's.
    Hit,
    /// B: it was another monster's.
    Miss,
    /// B: Link left the room first, an eligible carrier alive.
    Left,
    /// B: the window's conditions ended with no death pending.
    Closed,
    /// Either: Link entered Dodongo's room (or Dodongo appeared) with it open.
    Suspended,
    /// Either: still open when the replay ended.
    OpenAtEnd,
}

impl Outcome {
    pub const fn name(self) -> &'static str {
        match self {
            Outcome::Armed => "armed",
            Outcome::Missed => "missed",
            Outcome::Other => "other",
            Outcome::Lost => "lost",
            Outcome::PausedOut => "paused-out",
            Outcome::Hit => "hit",
            Outcome::Miss => "miss",
            Outcome::Left => "left",
            Outcome::Closed => "closed",
            Outcome::Suspended => "suspended",
            Outcome::OpenAtEnd => "open-at-end",
        }
    }

    /// An opportunity the metric counts.
    pub const fn opportunity(self) -> bool {
        matches!(self, Outcome::Armed | Outcome::Missed | Outcome::Other | Outcome::Hit | Outcome::Miss | Outcome::Left)
    }
}

/// The bombs a closed window brought: Metric A's forced drop after `armed`, Metric B's table drop after `hit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropFate {
    /// A hit's death dropped no bombs.
    NoBombs,
    /// Armed, not fallen yet (the next drop-capable death takes it).
    Pending,
    /// On the floor (the replay ended before its fate).
    Fell,
    Taken,
    /// Expired, or left on the floor.
    Wasted,
    /// Armed, then harm (or the fairy at 16 kills) took it before it fell.
    Lost,
}

impl DropFate {
    pub const fn name(self) -> &'static str {
        match self {
            DropFate::NoBombs => "none",
            DropFate::Pending => "pending",
            DropFate::Fell => "fell",
            DropFate::Taken => "taken",
            DropFate::Wasted => "wasted",
            DropFate::Lost => "lost",
        }
    }
}

/// One window of either metric, from the frame it opened to the frame it closed (frames: the states' `frame`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub metric: Metric,
    pub open_frame: u64,
    pub open_room: MapLocation,
    pub close_frame: u64,
    /// The room the closing step started in (a `left` window: the room left).
    pub close_room: MapLocation,
    pub outcome: Outcome,
    /// (type, slot) of what closed it: the 10th kill's victim, the death that moved the cycle.
    pub killed: Option<(u8, u8)>,
    /// The bag was full. Metric A: on the frame before the resolving action (before the bomb was placed, before the
    /// other weapon's 10th kill landed): the bag Link chose with. With a full bag a bomb 10th kill goes 8 → 7 → 8
    /// (the forced four capped at the bag): nothing gained, and the 5 rupees lost (the owner, Nov 23). Metric B: at
    /// the close (the kill does not change the bag; the carrier's drop comes after).
    pub full_bag: bool,
    /// The bombs it brought (`armed`, `hit`), else None.
    pub drop: Option<DropFate>,
}

/// Dropped bombs followed to their fate, each for its window (as `crate::bombs::BombStats` follows every bomb drop):
/// taken (gone while the lifetime could not have run out, or Link's count rose) or wasted (expired, or left on the
/// floor).
#[derive(Clone, Debug, Default)]
struct Followed(Vec<(i8, u8, usize)>);

impl Followed {
    fn add(&mut self, state: &GameView, slot: i8, window: usize) -> bool {
        let found = state.items.iter().find(|i| i.index == slot && i.id == item::BOMBS);
        if let Some(i) = found {
            self.0.push((slot, i.timer, window));
        }
        found.is_some()
    }

    fn update(&mut self, step: &Step, windows: &mut [Window]) {
        let (p, c) = (step.prev, step.curr);
        let reach = step.frames / 2 + crate::hearts::TIMER_MARGIN;
        let gained = c.bombs() > p.bombs();
        self.0.retain_mut(|(slot, timer, window)| {
            let now = c.items.iter().find(|i| i.index == *slot && i.id == item::BOMBS && i.timer <= *timer);
            if let Some(i) = now.filter(|_| !step.left) {
                *timer = i.timer;
                return true;
            }
            let taken = !step.left && (u32::from(*timer) > reach || gained);
            windows[*window].drop = Some(if taken { DropFate::Taken } else { DropFate::Wasted });
            false
        });
    }
}

/// Metric A's window over a run: opened at a step that starts inside [`tenth_bomb_window`], paused while Link is
/// on 9 without a bomb of his own in the bag or in the air (or the rule masks BOMB), resumed as the same window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Tenth {
    #[default]
    Closed,
    Open { frame: u64, room: MapLocation },
    Paused { frame: u64, room: MapLocation },
}

/// A bomb placed while Metric A's window opened at frame `id` was open, live until its animation slot goes
/// inactive; `full`: the bag was full on the frame before it was placed.
#[derive(Clone, Copy, Debug)]
struct LiveBomb {
    kind: AnimationKind,
    id: u64,
    full: bool,
    opened_wall: bool,
}

/// Counts of a list of windows (tests and quick looks; reports sum the list in Python): Metric A's opportunities are
/// `tenth_armed + tenth_missed + tenth_other`, Metric B's `carrier_hits + carrier_miss_kill + carrier_miss_leave`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BombWindowStats {
    pub tenth_armed: u32,
    pub tenth_missed: u32,
    pub tenth_other: u32,
    /// The three above with a full bag ([`Window::full_bag`]).
    pub tenth_full_armed: u32,
    pub tenth_full_missed: u32,
    pub tenth_full_other: u32,
    pub tenth_lost: u32,
    pub tenth_paused_out: u32,
    /// Armed drops that fell (taken, wasted or still on the floor), were taken, wasted, lost before falling.
    pub tenth_forced_fell: u32,
    pub tenth_forced_taken: u32,
    pub tenth_forced_wasted: u32,
    pub tenth_forced_lost: u32,
    pub carrier_hits: u32,
    pub carrier_miss_kill: u32,
    pub carrier_miss_leave: u32,
    pub carrier_closed: u32,
    /// Hits whose death dropped bombs, and those taken.
    pub carrier_bombs: u32,
    pub carrier_bombs_taken: u32,
    /// Hits and misses no slot accounted for (expected 0).
    pub carrier_unattributed: u32,
    pub suspended: u32,
    pub open_at_end: u32,
}

impl BombWindowStats {
    pub fn of(windows: &[Window]) -> BombWindowStats {
        let mut s = BombWindowStats::default();
        for w in windows {
            let full = u32::from(w.full_bag);
            let fell = matches!(w.drop, Some(DropFate::Fell | DropFate::Taken | DropFate::Wasted));
            let (taken, wasted) = (w.drop == Some(DropFate::Taken), w.drop == Some(DropFate::Wasted));
            match w.outcome {
                Outcome::Armed => {
                    s.tenth_armed += 1;
                    s.tenth_full_armed += full;
                    s.tenth_forced_fell += u32::from(fell);
                    s.tenth_forced_taken += u32::from(taken);
                    s.tenth_forced_wasted += u32::from(wasted);
                    s.tenth_forced_lost += u32::from(w.drop == Some(DropFate::Lost));
                }
                Outcome::Missed => {
                    s.tenth_missed += 1;
                    s.tenth_full_missed += full;
                }
                Outcome::Other => {
                    s.tenth_other += 1;
                    s.tenth_full_other += full;
                }
                Outcome::Lost => s.tenth_lost += 1,
                Outcome::PausedOut => s.tenth_paused_out += 1,
                Outcome::Hit => {
                    s.carrier_hits += 1;
                    s.carrier_bombs += u32::from(fell);
                    s.carrier_bombs_taken += u32::from(taken);
                }
                Outcome::Miss => s.carrier_miss_kill += 1,
                Outcome::Left => s.carrier_miss_leave += 1,
                Outcome::Closed => s.carrier_closed += 1,
                Outcome::Suspended => s.suspended += 1,
                Outcome::OpenAtEnd => s.open_at_end += 1,
            }
            s.carrier_unattributed += u32::from(matches!(w.outcome, Outcome::Hit | Outcome::Miss) && w.killed.is_none());
        }
        s
    }
}

/// Both windows over one run: [`BombWindows::frame`] on every counted frame ([`FrameKind::Play`]),
/// [`BombWindows::skip`] on the others, [`BombWindows::finish`] at the end; the closed windows in `windows`.
#[derive(Clone, Debug, Default)]
pub struct BombWindows {
    pub windows: Vec<Window>,
    tenth: Tenth,
    bombs: Vec<LiveBomb>,
    carrier: Option<(u64, MapLocation)>,
    forced: Followed,
    carried: Followed,
    /// The `armed` window whose forced drop has not fallen yet (Dodongo's death arms one too, zelda-asm
    /// `src/bank4/dodongo.asm`: not the metric's).
    armed_pending: Option<usize>,
    /// The last counted frame and Link's own bombs then.
    prev: Option<(Arc<GameView>, i32)>,
    /// A transition frame since `prev`, and frames since it.
    left: bool,
    gap: u32,
    /// Dodongo's room Link stands in: nothing counts until he leaves it.
    excluded: bool,
    /// Link died: nothing counts after.
    over: bool,
}

impl BombWindows {
    /// (Metric A's window unresolved, open or paused; Metric B's open).
    pub fn open(&self) -> (bool, bool) {
        (self.tenth != Tenth::Closed, self.carrier.is_some())
    }

    /// A frame not counted: a transition (`left`) ends Link's stay in the room.
    pub fn skip(&mut self, left: bool) {
        self.gap += 1;
        self.left |= left;
    }

    /// Counts the step from the last counted frame to `state` (`granted`: the cheated bombs in its bag, see
    /// [`own_bombs`]); `rules` answer for the step's start.
    pub fn frame(&mut self, state: Arc<GameView>, granted: i32, rules: &mut dyn WindowRules) {
        if self.over {
            return;
        }
        let own = own_bombs(&state, granted);
        if let Some((prev, own_prev)) = self.prev.take() {
            let left = self.left || prev.full_location() != state.full_location();
            let step = Step { prev: &prev, curr: &state, left, frames: self.gap + 1, own_prev };
            let dodongo = dodongo_here(&state);
            if self.excluded {
                // The step that leaves Dodongo's room is his room's too.
                self.excluded = !left;
            } else if !dodongo || left {
                // The step that enters his room is the room left's.
                self.update(&step, rules);
            }
            if dodongo && !self.excluded {
                self.suspend(&step);
            }
        } else {
            self.excluded = dodongo_here(&state);
        }
        (self.left, self.gap) = (false, 0);
        self.over = state.game_over();
        self.prev = Some((state, own));
    }

    /// The replay ended: open windows are `open-at-end`.
    pub fn finish(&mut self) {
        let Some((last, _)) = self.prev.clone() else { return };
        let end = Step { prev: &last, curr: &last, left: false, frames: 0, own_prev: 0 };
        let full = bag_full(&last);
        if let Tenth::Open { frame, room } | Tenth::Paused { frame, room } = self.tenth {
            self.close(Metric::Tenth, (frame, room), &end, Outcome::OpenAtEnd, None, full);
        }
        if let Some(open) = self.carrier.take() {
            self.close(Metric::Carrier, open, &end, Outcome::OpenAtEnd, None, full);
        }
        self.tenth = Tenth::Closed;
    }

    fn close(
        &mut self,
        metric: Metric,
        open: (u64, MapLocation),
        step: &Step,
        outcome: Outcome,
        killed: Option<(u8, u8)>,
        full_bag: bool,
    ) -> usize {
        self.windows.push(Window {
            metric,
            open_frame: open.0,
            open_room: open.1,
            close_frame: step.curr.frame,
            close_room: step.prev.full_location(),
            outcome,
            killed,
            full_bag,
            drop: None,
        });
        self.windows.len() - 1
    }

    /// Dodongo's room: the open windows are suspended (the bombs in the air go with them).
    fn suspend(&mut self, step: &Step) {
        let full = bag_full(step.prev);
        if let Tenth::Open { frame, room } | Tenth::Paused { frame, room } = self.tenth {
            self.close(Metric::Tenth, (frame, room), step, Outcome::Suspended, None, full);
        }
        if let Some(open) = self.carrier.take() {
            self.close(Metric::Carrier, open, step, Outcome::Suspended, None, full);
        }
        self.tenth = Tenth::Closed;
        self.bombs.clear();
        self.excluded = true;
    }

    /// Counts one step.
    pub fn update(&mut self, step: &Step, rules: &mut dyn WindowRules) {
        self.update_tenth(step, rules);
        self.update_carrier(step, rules);
    }

    fn update_tenth(&mut self, step: &Step, rules: &mut dyn WindowRules) {
        let (p, c) = (step.prev, step.curr);
        let window = p.next_kill_arms_drop() && step.own_prev >= 1 && rules.tenth_rule(p);
        self.tenth = match self.tenth {
            Tenth::Closed if window => Tenth::Open { frame: p.frame, room: p.full_location() },
            Tenth::Paused { frame, room } if window => Tenth::Open { frame, room },
            // Still on 9 (harm is closed within its own step), a bomb of the window in the air: still open.
            Tenth::Open { frame, room } if !window && p.next_kill_arms_drop() => {
                if self.bombs.iter().any(|b| b.id == frame) {
                    Tenth::Open { frame, room }
                } else {
                    Tenth::Paused { frame, room }
                }
            }
            Tenth::Open { frame, room } if !window => {
                self.close(Metric::Tenth, (frame, room), step, Outcome::Lost, None, bag_full(p));
                Tenth::Closed
            }
            t => t,
        };
        if let Tenth::Open { frame, .. } = self.tenth {
            let placed: Vec<AnimationKind> = step.placed_bombs().collect();
            let full = bag_full(p);
            self.bombs.extend(placed.into_iter().map(|kind| LiveBomb { kind, id: frame, full, opened_wall: false }));
        }
        let opened = opened_wall(p, c);
        for b in &mut self.bombs {
            b.opened_wall |= opened;
        }
        if step.tenth_kill() {
            let killed = new_kills(step).first().copied();
            match self.tenth {
                Tenth::Open { frame, room } => {
                    let armed = step.armed_bomb_drop();
                    // A bomb's kill: the bag before the first bomb of the window still in the air (the one that
                    // went off first); another weapon's: the bag on the frame before it landed.
                    let full = match self.bombs.iter().find(|b| b.id == frame) {
                        Some(b) if armed => b.full,
                        _ => bag_full(p),
                    };
                    let outcome = if armed { Outcome::Armed } else { Outcome::Other };
                    let index = self.close(Metric::Tenth, (frame, room), step, outcome, killed, full);
                    if armed {
                        self.windows[index].drop = Some(DropFate::Pending);
                        self.armed_pending = Some(index);
                    }
                }
                Tenth::Paused { frame, room } => {
                    self.close(Metric::Tenth, (frame, room), step, Outcome::PausedOut, killed, bag_full(p));
                }
                Tenth::Closed => {}
            }
            self.tenth = Tenth::Closed;
            self.bombs.clear();
        } else if self.tenth != Tenth::Closed && !c.next_kill_arms_drop() && !c.game_over() {
            if let Tenth::Open { frame, room } | Tenth::Paused { frame, room } = self.tenth {
                let outcome = if matches!(self.tenth, Tenth::Paused { .. }) { Outcome::PausedOut } else { Outcome::Lost };
                self.close(Metric::Tenth, (frame, room), step, outcome, None, bag_full(p));
            }
            self.tenth = Tenth::Closed;
            self.bombs.clear();
        }
        let current = match self.tenth {
            Tenth::Open { frame, room } | Tenth::Paused { frame, room } => Some((frame, room)),
            Tenth::Closed => None,
        };
        let mut missed = None;
        self.bombs.retain(|b| {
            let ended = step.left || c.animation_state(b.kind) == AnimationState::Inactive;
            if ended && !b.opened_wall && current.is_some_and(|(id, _)| id == b.id) {
                missed = Some(b.full);
            }
            !ended
        });
        if let (Some(full), Some(open)) = (missed, current) {
            self.close(Metric::Tenth, open, step, Outcome::Missed, None, full);
            self.tenth = Tenth::Closed;
            self.bombs.clear();
        }
        self.forced.update(step, &mut self.windows);
        let armed = |s: &GameView| s.help_drop_count() >= 10 && s.help_drop_is_bombs();
        if let Some(index) = self.armed_pending
            && !armed(c)
        {
            self.armed_pending = None;
            self.windows[index].drop = Some(DropFate::Lost);
            for slot in step.new_bomb_drops() {
                if self.forced.add(c, slot, index) {
                    self.windows[index].drop = Some(DropFate::Fell);
                }
            }
        }
    }

    fn update_carrier(&mut self, step: &Step, rules: &mut dyn WindowRules) {
        let (p, c) = (step.prev, step.curr);
        let window = carrier_window_with(p, step.own_prev, rules);
        if let Some(open) = self.carrier
            && !window
            && !death_pending(p)
        {
            self.close(Metric::Carrier, open, step, Outcome::Closed, None, bag_full(p));
            self.carrier = None;
        }
        if self.carrier.is_none() && window {
            self.carrier = Some((p.frame, p.full_location()));
        }
        self.carried.update(step, &mut self.windows);
        let (moves, deaths) = cycle_deaths(step);
        let Some(open) = self.carrier else { return };
        if moves > 0 {
            let hit = deaths.iter().find(|d| carrier_type(d.kind, d.slot));
            let killed = hit.or(deaths.first()).map(|d| (d.kind, d.slot));
            let outcome = if hit.is_some() { Outcome::Hit } else { Outcome::Miss };
            let index = self.close(Metric::Carrier, open, step, outcome, killed, bag_full(p));
            if hit.is_some() {
                self.windows[index].drop = Some(DropFate::NoBombs);
                for d in deaths.iter().filter(|d| carrier_type(d.kind, d.slot) && d.dropped == Some(item::BOMBS)) {
                    if self.carried.add(c, d.slot as i8, index) {
                        self.windows[index].drop = Some(DropFate::Fell);
                    }
                }
            }
            self.carrier = None;
        } else if step.left {
            self.close(Metric::Carrier, open, step, Outcome::Left, None, bag_full(p));
            self.carrier = None;
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

    /// [`bomb_carrier_window`] of `ram`, its whole bag Link's own.
    fn carrier(ram: &Ram, bomb_rule: bool, kill_all: bool) -> bool {
        let v = view(ram, 0);
        bomb_carrier_window(&v, v.bombs(), bomb_rule, kill_all)
    }

    /// [`tenth_bomb_window`] of `ram`, its whole bag Link's own.
    fn tenth(ram: &Ram, bomb_rule: bool) -> bool {
        let v = view(ram, 0);
        tenth_bomb_window(&v, v.bombs(), bomb_rule)
    }

    /// Feeds `rams` as consecutive play frames (no cheated bombs); the counts of the windows closed.
    fn play(w: &mut BombWindows, rams: &[Ram], bomb_rule: bool, kill_all: bool) -> BombWindowStats {
        let mut rules = FixedRules { bomb_rule, kill_all };
        for (i, ram) in rams.iter().enumerate() {
            w.frame(view(ram, 100 + i as u64), 0, &mut rules);
        }
        BombWindowStats::of(&w.windows)
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
        assert!(tenth(&base(1, 9, 3), true));
        assert!(!tenth(&base(1, 9, 3), false), "bombs: never");
        assert!(!tenth(&base(0, 9, 3), true), "no bomb");
        assert!(!tenth(&base(4, 8, 3), true), "not the 10th");
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
        assert!(carrier(&start, true, false));
        let mut killed = start;
        kill(&mut killed, 2, Some(item::BOMBS));
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, killed], true, false);
        assert_eq!((stats.carrier_hits, stats.carrier_miss_kill, stats.carrier_bombs), (1, 0, 1));
        assert!(!w.open().1);
        // The cycle now reads 6: no bomb column next.
        assert!(!carrier(&killed, true, false));
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
        assert!(!carrier(&darknuts, true, true), "FIGHT, Darknuts only");
        assert!(carrier(&darknuts, true, false), "pass-through");
        let mut mixed = darknuts;
        monster(&mut mixed, 4, enemy::STALFOS);
        assert!(carrier(&mixed, true, true));
        let shut = |f: &dyn Fn(&mut Ram)| {
            let mut ram = mixed;
            f(&mut ram);
            !carrier(&ram, true, false)
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
        assert!(!carrier(&gibdo, true, false));
        monster(&mut gibdo, 2, enemy::GIBDO);
        assert!(carrier(&gibdo, true, true), "slot 2's Gibdo, the slot-1 one the choice");
    }

    /// B: Bubbles are no monsters: Red Wizzrobes with Bubbles in a FIGHT room never count.
    #[test]
    fn bubbles_are_ignored() {
        let mut ram = base(2, 0, 0);
        monster(&mut ram, 2, enemy::RED_WIZZROBE);
        monster(&mut ram, 3, enemy::BUBBLE);
        monster(&mut ram, 4, enemy::BLUE_BUBBLE);
        assert!(!carrier(&ram, true, true));
        assert!(carrier(&ram, true, false));
    }

    /// B: a Vire is a carrier only with the magic sword or BOMB allowed with a bomb in the bag (else it splits).
    #[test]
    fn vire_needs_the_magic_sword_or_bombs() {
        let mut ram = base(0, 0, 0);
        monster(&mut ram, 2, enemy::VIRE);
        set(&mut ram, Var::Sword, 1);
        assert!(!carrier(&ram, true, false), "wood sword, no bombs");
        set(&mut ram, Var::Bombs, 2);
        assert!(!carrier(&ram, false, false), "bombs masked");
        assert!(carrier(&ram, true, false), "bombs allowed");
        set(&mut ram, Var::Bombs, 0);
        set(&mut ram, Var::Sword, 3);
        assert!(carrier(&ram, false, false), "magic sword");
        // A splitting Vire is no choice either: Vire and Stalfos under FIGHT with the wood sword and no bombs.
        set(&mut ram, Var::Sword, 1);
        monster(&mut ram, 3, enemy::STALFOS);
        assert!(!carrier(&ram, true, true));
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
        let mut w = BombWindows::default();
        w.frame(view(&ram, 0), 0, &mut rules);
        w.skip(false);
        w.frame(view(&ram, 2), 0, &mut rules);
        assert_eq!((BombWindowStats::of(&w.windows).carrier_miss_leave, w.open().1), (0, true));
        w.skip(true);
        w.frame(view(&ram, 4), 0, &mut rules);
        assert_eq!(BombWindowStats::of(&w.windows).carrier_miss_leave, 1);
    }

    /// B: while a death sparkle runs, the cycle speaks for a kill already made: no window opens, and that death
    /// counts for nothing.
    #[test]
    fn no_window_over_a_pending_death() {
        let mut start = base(2, 3, 0);
        monster(&mut start, 2, enemy::RED_DARKNUT);
        monster(&mut start, 3, enemy::STALFOS);
        start[tables::OBJ_SPAWN_STATE.0 + 3] = 0x10;
        assert!(!carrier(&start, true, false));
        let mut gone = start;
        gone[tables::OBJ_SPAWN_STATE.0 + 3] = 0;
        died(&mut gone, 3, None);
        let stats = play(&mut BombWindows::default(), &[start, gone], true, false);
        assert_eq!((stats.carrier_hits, stats.carrier_miss_kill), (0, 0));
    }

    /// A: with a full bag, a bomb 10th kill reads full (the bag before the bomb was placed: 8), though the bag is 7
    /// when it lands; the record names the window's frames, rooms and the killed Stalfos.
    #[test]
    fn full_bag_before_the_bomb_and_the_record() {
        let mut start = base(8, 9, 3);
        monster(&mut start, 2, enemy::STALFOS);
        let s1 = placed(&start);
        let mut s2 = bomb(&s1, 0);
        s2[tables::OBJ_SPAWN_STATE.0 + 2] = 0x10;
        set(&mut s2, Var::HelpDropCount, 10);
        set(&mut s2, Var::HelpDropValue, 1);
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, start, s1, s2], true, false);
        assert_eq!((stats.tenth_armed, stats.tenth_full_armed), (1, 1));
        let r = w.windows[0];
        assert_eq!((r.metric, r.outcome, r.open_frame, r.close_frame), (Metric::Tenth, Outcome::Armed, 100, 103));
        assert_eq!((r.close_room.value, r.killed, r.drop), (0x44, Some((enemy::STALFOS, 2)), Some(DropFate::Pending)));
    }

    /// Dodongo's room is outside both metrics: an open window is suspended on entering, nothing opens or closes
    /// there, and after he is left a new window may open.
    #[test]
    fn dodongo_rooms_are_excluded() {
        let mut start = base(2, 9, 0);
        monster(&mut start, 3, enemy::STALFOS);
        let mut dodongo = base(2, 9, 0);
        set(&mut dodongo, Var::Location, 0x45);
        monster(&mut dodongo, 1, enemy::DODONGO);
        monster(&mut dodongo, 3, enemy::STALFOS);
        let mut killed = dodongo;
        kill(&mut killed, 3, None);
        let mut out = base(2, 0, 0);
        set(&mut out, Var::Location, 0x46);
        monster(&mut out, 2, enemy::BLUE_MOBLIN);
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, start, dodongo, killed, killed, out, out], true, false);
        assert_eq!((stats.suspended, stats.tenth_other, stats.tenth_lost), (1, 0, 0), "{:?}", w.windows);
        assert!(w.open().1, "a carrier window opened after Dodongo's room");
    }

    /// A child Gel (`$14`) moves neither the cycle nor the room count; a Zol (`$13`) does (asm:Obj_EndMetastate).
    /// Both dying in one step move the cycle by one, and the death is the Zol's (a miss, not a carrier); the Gel
    /// is no monster of the choice either.
    #[test]
    fn a_child_gel_death_moves_nothing_beside_a_zol() {
        let mut start = base(2, 3, 0);
        monster(&mut start, 2, enemy::BLUE_MOBLIN);
        monster(&mut start, 3, enemy::GREEN_GEL);
        monster(&mut start, 4, enemy::ZOL);
        let mut gel_only = start;
        monster(&mut gel_only, 4, 0);
        assert!(!carrier(&gel_only, true, true), "a child Gel gives no choice in a FIGHT");
        let mut both = start;
        both[ITEM_OBJ_MONSTER_TYPE as usize + 3] = enemy::GREEN_GEL;
        both[tables::OBJ_ID.0 + 3] = 0;
        died(&mut both, 4, Some(item::RUPEE));
        let (p, c) = (view(&start, 0), view(&both, 8));
        let step = Step { prev: &p, curr: &c, left: false, frames: 8, own_prev: 2 };
        assert_eq!(cycle_deaths(&step), (1, vec![Death { slot: 4, kind: enemy::ZOL, dropped: Some(item::RUPEE) }]));
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, both], true, true);
        assert_eq!((stats.carrier_hits, stats.carrier_miss_kill), (0, 1));
        assert_eq!(w.windows[0].killed, Some((enemy::ZOL, 4)));
    }

    /// The frames of the 2.0.0.13 250 test's seed 49 at frame 23677 (4:0x31): on 9 kills, Link's first bomb is
    /// exploding, he places his last (the bag 1 → 0), and the first bomb's blast kills a Red Keese: `$50` 9 → 10, `$51`
    /// set. The window stays open with an empty bag while a bomb of it is in the air, and closes `armed`: the bomb
    /// that killed was placed with bombs in the bag. (A reference that asks for a bomb in the bag on the frame before
    /// the kill misses it.)
    #[test]
    fn the_last_bomb_in_the_air_keeps_the_window_open() {
        let mut start = base(2, 9, 4);
        monster(&mut start, 11, enemy::RED_KEESE);
        let first = placed(&start);
        let exploding = bomb(&first, BOMB_EXPLODED);
        let mut last = exploding;
        set(&mut last, Var::Bombs, 0);
        set(&mut last, Var::BombOrFlameAnimation2, BOMB_ACTIVE);
        let mut killed = last;
        killed[tables::OBJ_SPAWN_STATE.0 + 11] = 0x10;
        set(&mut killed, Var::HelpDropCount, 10);
        set(&mut killed, Var::HelpDropValue, 1);
        set(&mut killed, Var::KillStreak, 10);
        let mut w = BombWindows::default();
        let stats = play(&mut w, &[start, start, first, exploding, last, killed], true, false);
        assert_eq!((stats.tenth_armed, stats.tenth_paused_out, stats.tenth_full_armed), (1, 0, 0), "{:?}", w.windows);
        assert_eq!(w.windows[0].killed, Some((enemy::RED_KEESE, 11)));
    }
}
