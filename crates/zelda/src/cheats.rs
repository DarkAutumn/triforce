//! Cheated items: the bombs, keys and rupees a room needs that Link may lack (Lee, Nov 8-9), shared by the
//! specialist env and the chain runner (`env::specialist_env`, `env::chain`).
//!
//! Counts stay natural except where a room *needs* an item ([`needs`]); there, whenever Link holds fewer than the
//! need (at the start in the room, on entering it and after every action) the count is topped up to it, and the
//! top-up is *granted* (outstanding).  The needs:
//! - bombs: 1 in a game.yaml `requires_bombs` room (Dodongo, the 0x2c rock wall);
//! - keys: 1 in a dungeon room whose route step leaves through a door that is still locked (the game's door tiles,
//!   not being opened), unless Link has the magic key or the room's treasure is a key he hasn't taken yet (that key
//!   comes first: the cheated one only appears once it is taken, and spent, bringing him back to 0);
//! - rupees: the game.yaml `requires_rupees` amount (a shop the route buys from; no room sets it yet), not while a
//!   rupee change is pending ($67D/$67E: the game adds and takes rupees one per frame).
//!
//! One rule for every item, in [`CheatLedger::settle`] order:
//! - every fall of the count is a use (`used`); a fall while some of the count is granted uses the grant first:
//!   those units are *cheated* uses (`cheated`; a grant merely given never counts);
//! - a natural rise (a pickup) repays the grant unit for unit (the count keeps them, as natural ones);
//! - leaving the room takes back what is still granted (never natural units);
//! - in the room Link is in now, a count below the need is topped up to it (the new top-up is granted).
//!
//! The env writes the new counts with `Session::apply_ram` and carries the rebuilt state on (mask, observation);
//! the action's `Change` keeps what the action itself did, so a grant is never a pickup to the critic.

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
    let room = state.full_location();
    let rupees_pending = state.var(Var::RupeesToAdd) != 0 || state.var(Var::RupeesToSubtract) != 0;
    ItemCounts {
        bombs: map.requires_bombs(room) as i32,
        keys: key_need(map, completion, prev, state),
        rupees: if rupees_pending { 0 } else { map.requires_rupees(room) as i32 },
    }
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

/// One item's count after the last settle (the ledger's own write included) and how much of it is granted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Held {
    count: i32,
    granted: i32,
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
}

impl CheatLedger {
    /// Starts counting in `room` with `count` of each item (nothing used yet).
    pub fn reset(&mut self, room: MapLocation, need: ItemCounts<i32>, count: ItemCounts<i32>) -> CheatStep {
        *self = CheatLedger {
            held: ItemCounts::from_fn(|item| Held {
                count: count[item],
                granted: 0,
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
                let cheated = fall.min(held.granted);
                held.granted -= cheated;
                step.used[item] = fall as u32;
                step.cheated[item] = cheated as u32;
            }
            // A natural pickup leaves the grant alone (Lee, F173): the grant is used first, and leaving takes it back.
            let mut target = now;
            if left {
                if held.granted > 0 {
                    step.left_with[item] = Some((now - held.granted).max(0) as u8);
                }
                target -= held.granted;
                held.granted = 0;
            }
            if target < need[item] {
                held.granted += need[item] - target;
                target = need[item];
            }
            held.count = target;
            step.set[item] = (target != now).then_some(target as u8);
        }
        self.used += step.used;
        self.cheated += step.cheated;
        self.room = Some(room);
        step
    }

    /// The granted (outstanding, unused) part of `item`'s count.
    pub fn granted(&self, item: CheatItem) -> i32 {
        self.held[item].granted
    }

    /// The granted (outstanding) part of every item's count: what a run continued from this state must
    /// know is not Link's own.
    pub fn grants(&self) -> ItemCounts<u8> {
        ItemCounts::from_fn(|item| self.held[item].granted.clamp(0, 255) as u8)
    }

    /// `state`'s count of `item` without the granted part.
    pub fn natural(&self, item: CheatItem, state: &GameView) -> i32 {
        state.var(item.var()) - self.granted(item)
    }

    /// [`CheatLedger::reset`] on the session's state, writing the grants; returns the state to carry on.
    /// `completion` is the episode's route state (its start step).
    pub fn begin<N: Nes>(&mut self, session: &mut Session<N>, completion: &GameCompletion) -> Arc<GameView> {
        let state = session.state().clone();
        let need = needs(session.game_map(), completion, None, &state);
        let step = self.reset(state.full_location(), need, Self::counts(&state));
        Self::write(session, step)
    }

    /// [`CheatLedger::begin`] on a state a run is continued from, where `granted` of each item's count was
    /// granted (and not used) when the state was taken: those units stay granted, so using them counts as
    /// cheated and leaving the room takes them back.
    pub fn resume<N: Nes>(
        &mut self,
        session: &mut Session<N>,
        completion: &GameCompletion,
        granted: ItemCounts<u8>,
    ) -> Arc<GameView> {
        let state = self.begin(session, completion);
        for item in CheatItem::ALL {
            let held = &mut self.held[item];
            held.granted = (held.granted + i32::from(granted[item])).min(held.count);
        }
        state
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
