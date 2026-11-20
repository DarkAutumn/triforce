//! Cheated items: the bombs, keys and rupees a room needs that Link may lack (Lee, Nov 8-9), shared by the
//! specialist env and the chain runner (`env::specialist_env`, `env::chain`).
//!
//! Counts stay natural except where a room *needs* an item ([`needs`]); there, whenever Link holds fewer than the
//! need (at the start in the room, on entering it and after every action) the count is topped up to it, and the
//! top-up is *granted* (outstanding).  The needs:
//! - bombs: 1 in a game.yaml `requires_bombs` room (Dodongo, the 0x2c rock wall), but none once what the room is
//!   bombed for is open (round 24: an overworld rock wall's secret, every bomb wall of a dungeon room: 0x2c on the
//!   leg's plain mountain way, 5:0x66 and 5:0x65 on Dungeon 5's way back; [`bomb_need`]);
//! - keys: 1 in a dungeon room whose route step leaves through a door that is still locked (the game's door tiles,
//!   not being opened), unless Link has the magic key or the room's treasure is a key he hasn't taken yet (that key
//!   comes first: the cheated one only appears once it is taken, and spent, bringing him back to 0);
//! - rupees: the game.yaml `requires_rupees` amount (round 26: the shops the route buys from, 0x44's and 0x34's
//!   caves) until a ware is bought there, not while a rupee change is pending ($67D/$67E: the game adds and takes
//!   rupees one per frame; [`rupee_need`]); in a dungeon room while the monster its arrows are for lives (an arrow a
//!   rupee a shot): Gohma's 6:0x1c (round 27), Ganon's 9:0x42 until he begins to die (round 29), 8:0x4c's Pols
//!   Voices (round 32: the reserve of 8 and a shot).
//!
//! A route step may grant too (routes.yaml `grant:`, round 32: d3first's bag of 8 bombs at Dungeon 5's entrance, the
//! owner's "we have the ability to cheat them to the character in game"): on entering the step the count is topped
//! up to the grant ([`CheatLedger::route_grant`]) and the units added are *route-granted*.  The two kinds are two
//! tallies of the ledger, never mixed: a room's grant (`granted`, [`CheatLedger::granted`]) and the route's
//! (`route`, [`CheatLedger::route_granted`]).  A route grant outlives the room: leaving takes back only a room's.
//! While a route grant of an item is outstanding, Link's count of it (his own and the route's) is the bag the route
//! set aside, and a room's need is met by the room's own grant alone (5:0x66's and 5:0x65's walls grant their bomb
//! as before, so the bag reaches the Olive Garden whole).
//!
//! One rule for every item, in [`CheatLedger::settle`] order:
//! - every fall of the count is a use (`used`); a fall uses a room's grant first, then a route's, then Link's own:
//!   granted units used are *cheated* uses (`cheated`), counted in the room where they are used (a grant merely
//!   given never counts; a route grant still outstanding when the run ends is not counted);
//! - a natural rise (a pickup) leaves both grants alone (F173): the grants are still used first;
//! - leaving the room takes back what is still granted by the room (never a route's units, never natural units);
//! - in the room Link is in now, a count below the need (while a route grant is outstanding: the room's grant below
//!   it) is topped up to it (the new top-up is granted by the room).
//!
//! The env writes the new counts with `Session::apply_ram` and carries the rebuilt state on (mask, observation);
//! the action's `Change` keeps what the action itself did, so a grant is never a pickup to the critic.
//!
//! A granted bomb opens the wall and does nothing else (round 26): where the bomb is for something it opens
//! ([`bomb_opens`]: every `requires_bombs` room but Dodongo's, whose bomb is the weapon), the envs offer BOMB only
//! in reach of what the objective leaves through while a room's grant is outstanding ([`CheatLedger::granted`];
//! `env::config::ItemsConfig::bomb_targets`), in every bomb mode and on the 10th kill too.  So a cheated bomb of a
//! room's grant in such a room is always a bomb laid at the route's wall.  A route grant is not bound so: it is the
//! weapon the route gives (the Olive Garden's), kept by the route's `bomb_reserve` until the fight it is for.

use std::ops::{AddAssign, Index, IndexMut};
use std::sync::Arc;

use crate::enums::{Direction, MapLocation};
use crate::game::GameView;
use crate::game_map::GameMap;
use crate::objectives::{FLAG_ITEM_TAKEN, GameCompletion};
use crate::room_state::room_flags;
use crate::{Nes, Session, Var};

/// The items a room can need (the closed set the ledger cheats).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CheatItem {
    Bombs,
    Keys,
    Rupees,
}

impl CheatItem {
    pub const ALL: [CheatItem; 3] = [CheatItem::Bombs, CheatItem::Keys, CheatItem::Rupees];

    /// The RAM count.
    pub const fn var(self) -> Var {
        match self {
            CheatItem::Bombs => Var::Bombs,
            CheatItem::Keys => Var::Keys,
            CheatItem::Rupees => Var::Rupees,
        }
    }

    /// `bombs`, `keys`, `rupees` (metric and JSON names: `cheated/<name>`, `cheated_<name>`).
    pub const fn name(self) -> &'static str {
        match self {
            CheatItem::Bombs => "bombs",
            CheatItem::Keys => "keys",
            CheatItem::Rupees => "rupees",
        }
    }
}

/// One value per [`CheatItem`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ItemCounts<T = u32> {
    pub bombs: T,
    pub keys: T,
    pub rupees: T,
}

impl<T> ItemCounts<T> {
    pub fn from_fn(mut f: impl FnMut(CheatItem) -> T) -> Self {
        ItemCounts {
            bombs: f(CheatItem::Bombs),
            keys: f(CheatItem::Keys),
            rupees: f(CheatItem::Rupees),
        }
    }
}

impl ItemCounts {
    /// Any item counts.
    pub fn any(&self) -> bool {
        CheatItem::ALL.iter().any(|&item| self[item] > 0)
    }
}

impl<T> Index<CheatItem> for ItemCounts<T> {
    type Output = T;
    fn index(&self, item: CheatItem) -> &T {
        match item {
            CheatItem::Bombs => &self.bombs,
            CheatItem::Keys => &self.keys,
            CheatItem::Rupees => &self.rupees,
        }
    }
}

impl<T> IndexMut<CheatItem> for ItemCounts<T> {
    fn index_mut(&mut self, item: CheatItem) -> &mut T {
        match item {
            CheatItem::Bombs => &mut self.bombs,
            CheatItem::Keys => &mut self.keys,
            CheatItem::Rupees => &mut self.rupees,
        }
    }
}

impl<T: AddAssign + Copy> AddAssign for ItemCounts<T> {
    fn add_assign(&mut self, other: Self) {
        for item in CheatItem::ALL {
            self[item] += other[item];
        }
    }
}

/// What the room Link is in needs of each item now (module doc).  `prev` is the room the action came from (None at
/// an episode or run start); `completion` is the episode's route state before it follows this step (the key need
/// asks it which edge the route step leaves through: [`GameCompletion::route_edge_exit`]).
pub fn needs(
    map: &GameMap,
    completion: &GameCompletion,
    prev: Option<MapLocation>,
    state: &GameView,
) -> ItemCounts<i32> {
    ItemCounts {
        bombs: bomb_need(map, state),
        keys: key_need(map, completion, prev, state),
        rupees: rupee_need(map, state),
    }
}

/// The game.yaml `requires_rupees` of Link's room (a shop's cave the route buys in, round 26: 0x44's arrows 80,
/// 0x34's food 60) until a ware is bought there (the overworld room's item flag `$10`, set by any purchase,
/// asm:Cave_State2TalkOrShopOrDoorCharge), and not while a rupee change is pending ($67D/$67E: the game adds and
/// takes rupees one per frame, so the price is still being taken). In a dungeon room the need lasts while the monster
/// its arrows are for lives (an arrow at a rupee a shot, asm:Arrow_Wield): a Link who comes with no more rupees than
/// the route's reserve could never shoot. Round 27: Gohma (6:0x1c, an arrow is its only weakness); round 29: Ganon
/// (9:0x42: one silver arrow kills him, and with none the room is sealed for good) until he begins to die; round 32:
/// a Pols Voice (8:0x4c: one arrow kills every one on its line; the route's reserve there is 8, which the purse
/// d3first brings, 1-10 rupees, seldom passes).
fn rupee_need(map: &GameMap, state: &GameView) -> i32 {
    let room = state.full_location();
    let pending = state.var(Var::RupeesToAdd) != 0 || state.var(Var::RupeesToSubtract) != 0;
    let flags = crate::room_entry::ROOM_FLAGS_OW as usize + usize::from(room.value & 0x7F);
    let done = if room.level == 0 {
        state.ram()[flags] & FLAG_ITEM_TAKEN != 0
    } else {
        let ganon = crate::objectives::ganon(state).is_some_and(|(_, phase)| phase != crate::objectives::GanonPhase::Dying);
        let pols = state.enemies.iter().any(|e| e.id == crate::enums::enemy::POLS_VOICE && !e.is_dying());
        crate::objectives::gohma(state).is_none() && !ganon && !pols
    };
    if pending || done { 0 } else { i32::from(map.requires_rupees(room)) }
}

/// Whether `room` is `requires_bombs` for something a bomb opens, whatever its state now: an overworld room with a
/// `bomb_secret` rock wall, a dungeon room with a bomb wall.  Not a room that needs its bomb as the weapon (Dodongo's
/// 2:0x0e: no wall).
pub fn bomb_opens(map: &GameMap, room: MapLocation) -> bool {
    map.requires_bombs(room)
        && map.get(room).is_some_and(|info| {
            if room.level == 0 { info.bomb_secret.is_some() } else { info.exits.values().any(|e| e.bomb_wall) }
        })
}

/// 1 in a `requires_bombs` room while it still has something the route bombs: an overworld room's secret rock wall
/// still closed, a dungeon room's bomb walls not all open (a room with none, Dodongo's, always needs its bomb).
pub fn bomb_need(map: &GameMap, state: &GameView) -> i32 {
    let room = state.full_location();
    if !map.requires_bombs(room) {
        return 0;
    }
    let Some(info) = map.get(room) else { return 1 };
    let open = if room.level == 0 {
        info.bomb_secret.is_some() && !state.ow_bomb_secret_closed()
    } else {
        info.exits.values().any(|e| e.bomb_wall) && state.bomb_walls().is_empty()
    };
    (!open) as i32
}

/// 1 when the route step leaves Link's dungeon room through a door that is still locked and no key treasure of
/// the room is left to take first; else 0.
fn key_need(map: &GameMap, completion: &GameCompletion, prev: Option<MapLocation>, state: &GameView) -> i32 {
    if state.level() == 0 || state.magic_key() {
        return 0;
    }
    let Some(direction) = completion.route_edge_exit(prev, state) else {
        return 0;
    };
    if !door_still_locked(state, direction) {
        return 0;
    }
    let room = state.full_location();
    let key_treasure_left = map.get(room).is_some_and(|r| r.treasure.as_deref() == Some("key"))
        && room_flags(state.ram(), room.value) & FLAG_ITEM_TAKEN == 0;
    (!key_treasure_left) as i32
}

/// `$54` TriggeredDoorCmd (6-8: a door is opening) and `$55` TriggeredDoorDir (its direction bit).
const TRIGGERED_DOOR_CMD: usize = 0x54;
const TRIGGERED_DOOR_DIR: usize = 0x55;

/// The door in `direction` is locked and not being opened: the game spends the key when the opening animation
/// starts, a few frames before the door tiles change (a key granted then would be a second one).
fn door_still_locked(state: &GameView, direction: Direction) -> bool {
    let ram = state.ram();
    let opening =
        (6..=8).contains(&ram[TRIGGERED_DOOR_CMD]) && ram[TRIGGERED_DOOR_DIR] & direction.value() != 0;
    state.is_door_locked(direction) && !opening
}

/// What one action (or the start) did to the items.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheatStep {
    /// Units used (every fall of the count: bombs placed, keys spent, rupees paid).
    pub used: ItemCounts,
    /// Granted units used.
    pub cheated: ItemCounts,
    /// The count to write, when the rule changed it.
    pub set: ItemCounts<Option<u8>>,
    /// On leaving a room with granted units outstanding: the count the room was left with, without them. The
    /// pre-entry snapshot of the next room was taken during the action, before this take-back, so a state saved
    /// from it gets these counts written in (F148).
    pub left_with: ItemCounts<Option<u8>>,
}

/// One item's count after the last settle (the ledger's own write included) and how much of it is granted, by a
/// room (`granted`) and by the route (`route`; module doc).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Held {
    count: i32,
    granted: i32,
    route: i32,
}

/// A ledger's outstanding grants, by kind: what a run continued from its state must know is not Link's own (a
/// replay state's context, a chain checkpoint).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Grants {
    /// Granted by the room Link is in (taken back on leaving it).
    pub room: ItemCounts<u8>,
    /// Granted by a route step (`grant:`), outstanding until used.
    pub route: ItemCounts<u8>,
}

/// An episode's or run's cheated-items bookkeeping (module doc).
#[derive(Clone, Debug, Default)]
pub struct CheatLedger {
    /// Units used so far (only `bombs` is reported: bombs placed).
    pub used: ItemCounts,
    /// Cheated units used so far.
    pub cheated: ItemCounts,
    held: ItemCounts<Held>,
    room: Option<MapLocation>,
    /// The furthest route step the run has reached ([`CheatLedger::follow_route`]: a step's grant is given the first
    /// time the run reaches it, never again on coming back).
    route_step: Option<usize>,
}

impl CheatLedger {
    /// Starts counting in `room` with `count` of each item (nothing used or granted yet).
    pub fn reset(&mut self, room: MapLocation, need: ItemCounts<i32>, count: ItemCounts<i32>) -> CheatStep {
        *self = CheatLedger {
            held: ItemCounts::from_fn(|item| Held {
                count: count[item],
                ..Held::default()
            }),
            room: Some(room),
            ..CheatLedger::default()
        };
        self.settle(room, need, count)
    }

    /// Follows one action to `count` of each item in `room`, which needs `need`.
    pub fn settle(&mut self, room: MapLocation, need: ItemCounts<i32>, count: ItemCounts<i32>) -> CheatStep {
        let left = self.room != Some(room);
        let mut step = CheatStep::default();
        for item in CheatItem::ALL {
            let held = &mut self.held[item];
            let now = count[item];
            if now < held.count {
                let fall = held.count - now;
                let room_part = fall.min(held.granted);
                let route_part = (fall - room_part).min(held.route);
                held.granted -= room_part;
                held.route -= route_part;
                step.used[item] = fall as u32;
                step.cheated[item] = (room_part + route_part) as u32;
            }
            // A natural pickup leaves the grants alone (Lee, F173): they are used first, and leaving takes the room's
            // back.
            let mut target = now;
            if left {
                if held.granted > 0 {
                    step.left_with[item] = Some((now - held.granted).max(0) as u8);
                }
                target -= held.granted;
                held.granted = 0;
            }
            // A route grant outstanding sets the bag aside: the room's need is met by the room's own grant alone.
            let have = if held.route > 0 { held.granted } else { target };
            if have < need[item] {
                held.granted += need[item] - have;
                target += need[item] - have;
            }
            held.count = target;
            step.set[item] = (target != now).then_some(target as u8);
        }
        self.used += step.used;
        self.cheated += step.cheated;
        self.room = Some(room);
        step
    }

    /// A route step's grant (`grant:`, [`crate::game_map::RouteStep::grant`]): each item's bag (its count without a
    /// room's grant) topped up to `grant`; the units added are route-granted (module doc).
    pub fn route_grant(&mut self, grant: ItemCounts<u8>) -> CheatStep {
        let mut step = CheatStep::default();
        for item in CheatItem::ALL {
            let held = &mut self.held[item];
            let add = i32::from(grant[item]) - (held.count - held.granted);
            if add > 0 {
                held.route += add;
                held.count += add;
                step.set[item] = Some(held.count.clamp(0, 255) as u8);
            }
        }
        step
    }

    /// The part of `item`'s count a room granted (outstanding, unused).
    pub fn granted(&self, item: CheatItem) -> i32 {
        self.held[item].granted
    }

    /// The part of `item`'s count a route step granted (outstanding, unused).
    pub fn route_granted(&self, item: CheatItem) -> i32 {
        self.held[item].route
    }

    /// Both kinds of outstanding grant of every item.
    pub fn grants(&self) -> Grants {
        let of = |part: fn(&Held) -> i32| ItemCounts::from_fn(|item| part(&self.held[item]).clamp(0, 255) as u8);
        Grants { room: of(|h| h.granted), route: of(|h| h.route) }
    }

    /// `state`'s count of `item` without the granted parts (Link's own).
    pub fn natural(&self, item: CheatItem, state: &GameView) -> i32 {
        state.var(item.var()) - self.granted(item) - self.route_granted(item)
    }

    /// [`CheatLedger::reset`] on the session's state, writing the room's grants, then the start step's route grant
    /// ([`CheatLedger::follow_route`]); returns the state to carry on.  `completion` is the episode's route state
    /// (its start step).
    pub fn begin<N: Nes>(&mut self, session: &mut Session<N>, completion: &GameCompletion) -> Arc<GameView> {
        let state = self.start(session, completion);
        self.follow_route(session, completion).unwrap_or(state)
    }

    /// [`CheatLedger::reset`] on a state a run is continued from, where `grants` of each item's count were granted
    /// (and not used) when the state was taken: those units stay granted, each by its kind (using them counts as
    /// cheated; leaving the room takes back the room's).  No route grant is given again: the run has reached its
    /// step already.
    pub fn resume<N: Nes>(&mut self, session: &mut Session<N>, completion: &GameCompletion, grants: Grants) -> Arc<GameView> {
        let state = self.start(session, completion);
        self.route_step = completion.step();
        for item in CheatItem::ALL {
            let held = &mut self.held[item];
            held.granted = (held.granted + i32::from(grants.room[item])).min(held.count);
            held.route = i32::from(grants.route[item]).min(held.count - held.granted);
        }
        state
    }

    fn start<N: Nes>(&mut self, session: &mut Session<N>, completion: &GameCompletion) -> Arc<GameView> {
        let state = session.state().clone();
        let need = needs(session.game_map(), completion, None, &state);
        let step = self.reset(state.full_location(), need, Self::counts(&state));
        Self::write(session, step)
    }

    /// [`CheatLedger::settle`] on the session's state after an action from room `prev`, writing the counts the rule
    /// sets; returns the step and the state to carry on (the action's `Change` keeps what the action itself did).
    /// `completion` hasn't followed the action yet.
    pub fn after_action<N: Nes>(
        &mut self,
        session: &mut Session<N>,
        completion: &GameCompletion,
        prev: MapLocation,
    ) -> (CheatStep, Arc<GameView>) {
        let state = session.state().clone();
        let need = needs(session.game_map(), completion, Some(prev), &state);
        let step = self.settle(state.full_location(), need, Self::counts(&state));
        (step, Self::write(session, step))
    }

    /// After `completion` has followed an action (or at a run's start): the first time the run reaches a route step
    /// with a `grant:`, [`CheatLedger::route_grant`] written; returns the state to carry on, None when nothing was
    /// granted (the grant is no pickup: the action's `Change` keeps what the action itself did).
    pub fn follow_route<N: Nes>(&mut self, session: &mut Session<N>, completion: &GameCompletion) -> Option<Arc<GameView>> {
        let step = completion.step()?;
        if self.route_step.is_some_and(|reached| reached >= step) {
            return None;
        }
        self.route_step = Some(step);
        let grant = completion.step_grant()?;
        let step = self.route_grant(grant);
        CheatItem::ALL.iter().any(|&item| step.set[item].is_some()).then(|| Self::write(session, step))
    }

    fn counts(state: &GameView) -> ItemCounts<i32> {
        ItemCounts::from_fn(|item| state.var(item.var()))
    }

    fn write<N: Nes>(session: &mut Session<N>, step: CheatStep) -> Arc<GameView> {
        let values: Vec<(Var, u8)> = CheatItem::ALL
            .into_iter()
            .filter_map(|item| step.set[item].map(|n| (item.var(), n)))
            .collect();
        if values.is_empty() {
            session.state().clone()
        } else {
            session.apply_ram(&values).clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DODONGO: MapLocation = MapLocation::new(2, 0x0e, false);
    const SOUTH: MapLocation = MapLocation::new(2, 0x1e, false);
    const SHOP: MapLocation = MapLocation::new(0, 0x10, true);
    const OUTSIDE: MapLocation = MapLocation::new(0, 0x10, false);

    fn of(bombs: i32, keys: i32, rupees: i32) -> ItemCounts<i32> {
        ItemCounts { bombs, keys, rupees }
    }

    fn set(bombs: Option<u8>, keys: Option<u8>, rupees: Option<u8>) -> ItemCounts<Option<u8>> {
        ItemCounts { bombs, keys, rupees }
    }

    fn cheats(bombs: u32, keys: u32, rupees: u32) -> ItemCounts {
        ItemCounts { bombs, keys, rupees }
    }

    const NONE: ItemCounts<i32> = ItemCounts { bombs: 0, keys: 0, rupees: 0 };
    const BOMB: ItemCounts<i32> = ItemCounts { bombs: 1, keys: 0, rupees: 0 };
    const KEY: ItemCounts<i32> = ItemCounts { bombs: 0, keys: 1, rupees: 0 };

    #[test]
    fn a_cheated_bomb_arms_counts_when_placed_and_rearms() {
        let mut l = CheatLedger::default();
        let s = l.reset(DODONGO, BOMB, of(0, 2, 7));
        assert_eq!(s.set, set(Some(1), None, None), "no bombs in the room: one is cheated in");
        assert_eq!((l.used, l.cheated), (cheats(0, 0, 0), cheats(0, 0, 0)), "granting doesn't count");
        assert_eq!(l.settle(DODONGO, BOMB, of(1, 2, 7)), CheatStep::default(), "nothing happened");
        let s = l.settle(DODONGO, BOMB, of(0, 2, 7));
        assert_eq!((s.used.bombs, s.cheated.bombs, s.set.bombs), (1, 1, Some(1)), "placed: counted, re-armed");
        assert_eq!((l.used, l.cheated), (cheats(1, 0, 0), cheats(1, 0, 0)), "keys and rupees untouched");
    }

    #[test]
    fn natural_bombs_count_as_used_only() {
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(DODONGO, BOMB, of(2, 0, 0)).set, set(None, None, None));
        assert_eq!(l.settle(DODONGO, BOMB, of(1, 0, 0)).used, cheats(1, 0, 0));
        // The natural last bomb: placing it is used, not cheated; then a cheated one arrives.
        let s = l.settle(DODONGO, BOMB, of(0, 0, 0));
        assert_eq!((s.used, s.cheated, s.set.bombs), (cheats(1, 0, 0), cheats(0, 0, 0), Some(1)));
        // A pickup leaves the grant alone (F173): the next bomb placed is the granted one, cheated.
        assert_eq!(l.settle(DODONGO, BOMB, of(5, 0, 0)).set, set(None, None, None));
        assert_eq!(l.settle(DODONGO, BOMB, of(4, 0, 0)).cheated, cheats(1, 0, 0));
        assert_eq!((l.used.bombs, l.cheated.bombs), (3, 1));
    }

    #[test]
    fn leaving_takes_back_only_what_is_still_granted() {
        let mut l = CheatLedger::default();
        l.reset(DODONGO, BOMB, of(0, 3, 9));
        let s = l.settle(SOUTH, NONE, of(1, 3, 9));
        assert_eq!((s.used, s.cheated, s.set), (cheats(0, 0, 0), cheats(0, 0, 0), set(Some(0), None, None)));
        // Outside, counts go up and down naturally.
        assert_eq!(l.settle(SOUTH, NONE, of(0, 3, 9)), CheatStep::default());
        assert_eq!(l.settle(SOUTH, NONE, of(3, 3, 9)), CheatStep::default());
        assert_eq!(l.settle(SOUTH, NONE, of(2, 2, 9)).used, cheats(1, 1, 0));
        assert_eq!(l.cheated, cheats(0, 0, 0));

        // A natural last bomb leaves with Link.
        let mut l = CheatLedger::default();
        l.reset(DODONGO, BOMB, of(1, 0, 0));
        assert_eq!(l.settle(SOUTH, NONE, of(1, 0, 0)), CheatStep::default());
        // A cheated bomb used up after a pickup: what is left is natural.
        let mut l = CheatLedger::default();
        l.reset(DODONGO, BOMB, of(0, 0, 0));
        for bombs in [4, 3, 2, 1] {
            l.settle(DODONGO, BOMB, of(bombs, 0, 0));
        }
        assert_eq!(l.settle(SOUTH, NONE, of(1, 0, 0)).set, set(None, None, None));
    }

    #[test]
    fn rooms_without_a_need_are_untouched() {
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(SOUTH, NONE, of(0, 0, 0)), CheatStep::default());
        assert_eq!(l.settle(SOUTH, NONE, of(0, 0, 0)), CheatStep::default());
        assert_eq!((l.used, l.cheated), (cheats(0, 0, 0), cheats(0, 0, 0)));
    }

    /// A key room (its route exit locked, its key treasure taken): with none Link gets one; spending it on a door
    /// counts one cheated key, and while the route door stays locked the floor re-arms. A natural key spent is used
    /// only. A need that drops (the route door opened) leaves the grant until Link leaves, which takes it back.
    #[test]
    fn a_cheated_key_counts_when_spent_and_is_taken_back_on_leaving() {
        const ROOM: MapLocation = MapLocation::new(1, 0x33, false);
        const NEXT: MapLocation = MapLocation::new(1, 0x23, false);
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(ROOM, KEY, of(0, 0, 0)).set, set(None, Some(1), None));
        assert_eq!(l.granted(CheatItem::Keys), 1);
        // Spent on the wrong door: cheated, and the route door is still locked: another one.
        let s = l.settle(ROOM, KEY, of(0, 0, 0));
        assert_eq!((s.used, s.cheated, s.set), (cheats(0, 1, 0), cheats(0, 1, 0), set(None, Some(1), None)));
        // The route door opened some other way: no need, the grant stays while Link is here...
        assert_eq!(l.settle(ROOM, NONE, of(0, 1, 0)), CheatStep::default());
        // ... and is taken back when he leaves.
        assert_eq!(l.settle(NEXT, NONE, of(0, 1, 0)).set, set(None, Some(0), None));
        assert_eq!((l.used.keys, l.cheated.keys), (1, 1));

        // A natural key (Link has one): no grant; spending it is used, then a cheated one arrives.
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(ROOM, KEY, of(0, 1, 0)).set, set(None, None, None));
        let s = l.settle(ROOM, KEY, of(0, 0, 0));
        assert_eq!((s.used.keys, s.cheated.keys, s.set.keys), (1, 0, Some(1)));
        // The key picked up in the room leaves the grant alone (F173): leaving takes the granted one back.
        assert_eq!(l.settle(ROOM, KEY, of(0, 2, 0)).set, set(None, None, None));
        assert_eq!(l.granted(CheatItem::Keys), 1);
        let s = l.settle(NEXT, NONE, of(0, 2, 0));
        assert_eq!((s.set, s.left_with), (set(None, Some(1), None), set(None, Some(1), None)));
    }

    /// The key-treasure-first rule as the ledger sees it: while the room's key treasure is left the need is 0 (no
    /// grant); once it is taken (count 0 → 1, natural) and spent (1 → 0) the need is 1 and the cheated key appears.
    #[test]
    fn the_rooms_key_treasure_comes_before_a_cheated_key() {
        const ROOM: MapLocation = MapLocation::new(1, 0x33, false);
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(ROOM, NONE, of(0, 0, 0)), CheatStep::default(), "treasure left: no key");
        assert_eq!(l.settle(ROOM, KEY, of(0, 1, 0)), CheatStep::default(), "the treasure key: natural");
        let s = l.settle(ROOM, KEY, of(0, 0, 0));
        assert_eq!((s.used.keys, s.cheated.keys, s.set.keys), (1, 0, Some(1)), "spent, then the cheated one");
        assert_eq!(l.granted(CheatItem::Keys), 1);
    }

    /// Rupees: topped up to the need; spending uses the top-up first (cheated), a natural rise leaves it alone
    /// (F173), and leaving takes back what is left of the top-up, never natural rupees.
    #[test]
    fn rupees_top_up_to_the_need_and_leave_natural_ones_alone() {
        let need = of(0, 0, 80);
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(SHOP, need, of(0, 0, 30)).set, set(None, None, Some(80)));
        assert_eq!(l.granted(CheatItem::Rupees), 50);
        // Paying 60 (the game takes them over several actions; mid-transfer the need is 0: no top-up).
        assert_eq!(l.settle(SHOP, NONE, of(0, 0, 50)).cheated, cheats(0, 0, 30));
        let s = l.settle(SHOP, NONE, of(0, 0, 20));
        assert_eq!((s.used.rupees, s.cheated.rupees, s.set.rupees), (30, 20, None), "the top-up first");
        assert_eq!(l.granted(CheatItem::Rupees), 0);
        // Settled at 20: topped up again (60 granted); a natural 5 leaves the grant as it is.
        assert_eq!(l.settle(SHOP, need, of(0, 0, 20)).set.rupees, Some(80));
        assert_eq!(l.settle(SHOP, need, of(0, 0, 85)), CheatStep::default());
        assert_eq!(l.granted(CheatItem::Rupees), 60);
        // Leaving takes back the 60 still granted: Link leaves with his 20 plus the 5 he found.
        assert_eq!(l.settle(OUTSIDE, NONE, of(0, 0, 85)).set, set(None, None, Some(25)));
        assert_eq!((l.used.rupees, l.cheated.rupees), (60, 50));
        // Natural rupees above the need are never touched.
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(SHOP, need, of(0, 0, 120)), CheatStep::default());
        assert_eq!(l.settle(SHOP, need, of(0, 0, 100)).set, set(None, None, None));
        assert_eq!(l.settle(OUTSIDE, NONE, of(0, 0, 100)), CheatStep::default());
        assert_eq!(l.cheated, cheats(0, 0, 0));
    }

    /// F173, the 0x2c example: no natural bomb, one granted, a 4-bomb pickup, the wall bomb placed: the grant is used
    /// first (1 cheated), and Link leaves with exactly his 4 picked-up bombs.
    #[test]
    fn a_pickup_leaves_the_grant_alone() {
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(DODONGO, BOMB, of(0, 0, 0)).set, set(Some(1), None, None));
        l.settle(DODONGO, BOMB, of(5, 0, 0));
        l.settle(DODONGO, BOMB, of(4, 0, 0));
        let s = l.settle(SOUTH, NONE, of(4, 0, 0));
        assert_eq!(s.set, set(None, None, None), "nothing left to take back");
        assert_eq!((l.used.bombs, l.cheated.bombs), (1, 1));
    }

    /// Round 26, a bomb-wall room entered with no bombs (7:0x19, its E wall the only one): the granted bomb laid at
    /// the wall is one cheated bomb; while its fuse burns the wall is still shut, so the floor re-arms; once the
    /// wall is open the room needs none, and the re-armed bomb, which the envs never offered anywhere, is taken back
    /// on leaving: one cheated bomb for the wall, and Link leaves with none.  (A room with a second bomb wall the
    /// route never opens, 7:0x69's W, keeps its need: the same, the bomb in hand is taken back on leaving.)
    #[test]
    fn a_grant_spent_at_the_wall_is_one_cheated_bomb() {
        const WALL: MapLocation = MapLocation::new(7, 0x19, false);
        const NEXT: MapLocation = MapLocation::new(7, 0x1a, false);
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(WALL, BOMB, of(0, 5, 0)).set.bombs, Some(1));
        let s = l.settle(WALL, BOMB, of(0, 5, 0));
        assert_eq!((s.used.bombs, s.cheated.bombs, s.set.bombs), (1, 1, Some(1)), "laid; the fuse burns: re-armed");
        assert_eq!(l.settle(WALL, NONE, of(1, 5, 0)), CheatStep::default(), "the wall is open: no need");
        assert_eq!(l.granted(CheatItem::Bombs), 1, "the re-armed bomb waits, granted");
        let s = l.settle(NEXT, NONE, of(1, 5, 0));
        assert_eq!((s.set.bombs, s.left_with.bombs), (Some(0), Some(0)), "taken back on leaving");
        assert_eq!((l.used.bombs, l.cheated.bombs), (1, 1));
        // Entered with the wall open (the way back): no need, no grant.
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(WALL, NONE, of(0, 5, 0)), CheatStep::default());
        assert_eq!(l.granted(CheatItem::Bombs), 0);
    }

    /// Round 32, d3first's bag: Dungeon 5's entrance tops the bag up to 8 (Link's 2 and 6 granted by the route).
    /// Leaving takes none back; the bag is used where it is thrown, the route's units first (cheated there) and then
    /// Link's own; a pickup leaves them alone; what is still granted when the run ends was never used, never counted.
    #[test]
    fn a_route_grant_outlives_the_room_and_counts_where_it_is_used() {
        const ENTRANCE: MapLocation = MapLocation::new(5, 0x76, false);
        const GARDEN: MapLocation = MapLocation::new(5, 0x05, false);
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(ENTRANCE, NONE, of(2, 6, 3)), CheatStep::default());
        let s = l.route_grant(ItemCounts { bombs: 8, keys: 0, rupees: 0 });
        assert_eq!((s.set, s.used, s.cheated), (set(Some(8), None, None), cheats(0, 0, 0), cheats(0, 0, 0)), "given");
        assert_eq!((l.granted(CheatItem::Bombs), l.route_granted(CheatItem::Bombs)), (0, 6));
        assert_eq!(l.grants(), Grants { room: ItemCounts::default(), route: ItemCounts { bombs: 6, keys: 0, rupees: 0 } });
        // Rooms change: nothing is taken back.
        assert_eq!(l.settle(SOUTH, NONE, of(8, 6, 3)), CheatStep::default());
        assert_eq!(l.settle(GARDEN, NONE, of(8, 6, 3)), CheatStep::default());
        // Three thrown in the garden: the route's, cheated there; a 4-bomb pickup leaves the grant as it is.
        let s = l.settle(GARDEN, NONE, of(5, 6, 3));
        assert_eq!((s.used, s.cheated), (cheats(3, 0, 0), cheats(3, 0, 0)));
        assert_eq!(l.settle(GARDEN, NONE, of(8, 6, 3)), CheatStep::default());
        assert_eq!(l.route_granted(CheatItem::Bombs), 3);
        // Five more: the route's last three, then two of Link's own.
        let s = l.settle(GARDEN, NONE, of(3, 6, 3));
        assert_eq!((s.used.bombs, s.cheated.bombs, l.route_granted(CheatItem::Bombs)), (5, 3, 0));
        assert_eq!((l.used.bombs, l.cheated.bombs), (8, 6));
        // A second grant tops up only what the bag lacks; unused at the end, it counts nothing.
        let s = l.route_grant(ItemCounts { bombs: 4, keys: 0, rupees: 0 });
        assert_eq!((s.set.bombs, l.route_granted(CheatItem::Bombs)), (Some(4), 1));
        assert_eq!(l.cheated.bombs, 6);
        // A bag already full is granted nothing.
        let mut l = CheatLedger::default();
        l.reset(ENTRANCE, NONE, of(8, 0, 0));
        assert_eq!(l.route_grant(ItemCounts { bombs: 8, keys: 0, rupees: 0 }), CheatStep::default());
        assert_eq!(l.route_granted(CheatItem::Bombs), 0);
    }

    /// The two kinds of grant apart, in a bomb-wall room crossed with the bag (5:0x66, its W wall shut): while the
    /// route's grant is outstanding the room's need is met by the room's own grant alone, so the wall gets its bomb
    /// as on a run without a bag (granted on entry, a 9th; laid; re-armed while the fuse burns; taken back on
    /// leaving) and the bag, Link's 2 and the route's 6, leaves the room whole.  Only the room's grant is what
    /// [`CheatLedger::granted`] reports (BOMB kept to the wall, `env::config::ItemsConfig::bomb_targets`).
    #[test]
    fn a_rooms_need_is_its_own_grant_while_the_route_sets_the_bag_aside() {
        const ENTRANCE: MapLocation = MapLocation::new(5, 0x76, false);
        const WALL: MapLocation = MapLocation::new(5, 0x66, false);
        const NEXT: MapLocation = MapLocation::new(5, 0x65, false);
        let mut l = CheatLedger::default();
        l.reset(ENTRANCE, NONE, of(2, 0, 0));
        l.route_grant(ItemCounts { bombs: 8, keys: 0, rupees: 0 });
        let s = l.settle(WALL, BOMB, of(8, 0, 0));
        assert_eq!(s.set.bombs, Some(9), "the wall's own bomb on top of the bag");
        assert_eq!((l.granted(CheatItem::Bombs), l.route_granted(CheatItem::Bombs)), (1, 6));
        let s = l.settle(WALL, BOMB, of(8, 0, 0));
        assert_eq!((s.used.bombs, s.cheated.bombs, s.set.bombs), (1, 1, Some(9)), "laid: the room's; re-armed");
        assert_eq!(l.settle(WALL, NONE, of(9, 0, 0)), CheatStep::default(), "the wall is open");
        let s = l.settle(NEXT, NONE, of(9, 0, 0));
        assert_eq!((s.set.bombs, s.left_with.bombs), (Some(8), Some(8)), "the room's taken back, the bag kept");
        assert_eq!((l.granted(CheatItem::Bombs), l.route_granted(CheatItem::Bombs)), (0, 6));
        assert_eq!((l.used.bombs, l.cheated.bombs), (1, 1));
        // Without a route grant a room's need is met by what Link holds, as ever.
        let mut l = CheatLedger::default();
        assert_eq!(l.reset(WALL, BOMB, of(2, 0, 0)), CheatStep::default());
        assert_eq!(l.settle(WALL, BOMB, of(1, 0, 0)).cheated.bombs, 0, "his own bomb at the wall");
    }

    /// A run continued from a state keeps each kind as it was ([`CheatLedger::grants`] round trip): leaving takes
    /// back the room's units only, the route's stay until used.
    #[test]
    fn grants_round_trip_by_kind() {
        const ROOM: MapLocation = MapLocation::new(5, 0x66, false);
        let mut l = CheatLedger::default();
        l.reset(ROOM, NONE, of(0, 0, 0));
        l.route_grant(ItemCounts { bombs: 8, keys: 0, rupees: 0 });
        l.settle(ROOM, BOMB, of(8, 0, 0));
        let grants = l.grants();
        assert_eq!((grants.room.bombs, grants.route.bombs), (1, 8));
        let mut m = CheatLedger::default();
        m.reset(ROOM, NONE, of(9, 0, 0));
        for item in CheatItem::ALL {
            m.held[item].granted = i32::from(grants.room[item]);
            m.held[item].route = i32::from(grants.route[item]);
        }
        assert_eq!(m.settle(SOUTH, NONE, of(9, 0, 0)).set.bombs, Some(8));
        assert_eq!(m.settle(SOUTH, NONE, of(6, 0, 0)).cheated.bombs, 2, "the route's");
    }

    /// [`bomb_opens`] on game.yaml: every `requires_bombs` room is bombed for something it opens (a dungeon bomb
    /// wall, an overworld rock wall) but Dodongo's 2:0x0e, where the bomb is the weapon; an untagged room is none.
    #[test]
    fn every_requires_bombs_room_but_dodongos_has_something_to_open() {
        let map = GameMap::builtin();
        let weapon: Vec<MapLocation> = map
            .rooms()
            .keys()
            .copied()
            .filter(|&room| map.requires_bombs(room) && !bomb_opens(&map, room))
            .collect();
        assert_eq!(weapon, [DODONGO]);
        for (level, room) in [(0, 0x2c), (0, 0x67), (3, 0x5b), (3, 0x4c), (3, 0x4d), (5, 0x66), (7, 0x69), (7, 0x1a)] {
            assert!(bomb_opens(&map, MapLocation::new(level, room, false)), "{level}:{room:#04x}");
        }
        assert!(!bomb_opens(&map, SOUTH), "2:0x1e is no requires_bombs room");
        assert!(!bomb_opens(&map, MapLocation::new(7, 0x1b, false)), "7:0x1b's W wall is how the route comes in");
    }

    /// Items are independent: a bomb room that is also a key room grants both and counts each separately.
    #[test]
    fn every_item_follows_the_same_rule_independently() {
        let mut l = CheatLedger::default();
        let need = of(1, 1, 10);
        assert_eq!(l.reset(DODONGO, need, of(0, 0, 0)).set, set(Some(1), Some(1), Some(10)));
        let s = l.settle(DODONGO, need, of(0, 1, 10));
        assert_eq!((s.cheated, s.set), (cheats(1, 0, 0), set(Some(1), None, None)));
        let s = l.settle(SOUTH, NONE, of(1, 1, 4));
        assert_eq!((s.cheated, s.set), (cheats(0, 0, 6), set(Some(0), Some(0), Some(0))), "all taken back");
        assert_eq!(l.cheated, cheats(1, 0, 6));
    }
}
