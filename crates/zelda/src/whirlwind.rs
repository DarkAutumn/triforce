//! The recorder's whirlwind on the overworld (round 26; `docs/routes/late-mechanics.md` §2.9, zelda-asm/items.md
//! "Recorder"): which way Link must face and how many times he must play for it to set him down in a given landing
//! room, read from the game's own counter.
//!
//! Every overworld play in mode 5 (but in Q1 room 0x42, the pond) first steps `TeleportingLevelIndex` (`$523`):
//! +1 facing right or up, -1 facing left or down, then on the same way past the levels whose triforce piece Link
//! lacks (asm:Whirlwind_Summon, asm:Whirlwind_StepLevelIndex). Only the first play of a room entry makes a whirlwind
//! (`IsWhirlwindSummoned` `$508`, cleared on entry); a later play before the pickup still steps the index, and the
//! destination is the index's level at the pickup (asm:Whirlwind_Update_Full). The index is never cleared but at
//! power on, so every earlier play counts: the plays are computed from it, never assumed.

use crate::enums::Direction;
use crate::game::GameView;

/// `TeleportingLevelIndex`: the whirlwind's destination level - 1 (used AND 7).
pub const LEVEL_INDEX: u16 = 0x523;
/// `IsWhirlwindSummoned`: a whirlwind was made in this room entry.
pub const SUMMONED: u16 = 0x508;
/// `WhirlwindTeleportingState`: 0 idle, 1 carrying Link away, 2 setting him down.
pub const TELEPORTING: u16 = 0x522;
/// `InvTriforce`: bit `level - 1` per piece.
const INV_TRIFORCE: u16 = 0x671;
/// Object type of the whirlwind (asm:Whirlwind_SetUp).
pub const OBJ_WHIRLWIND: u8 = 0x2E;
const OBJ_TYPE: u16 = 0x34F;

/// The index after one play facing up or right (`up`) or down or left, with `triforce` the pieces held
/// (asm:Whirlwind_Summon's loop). Without any piece the index steps once.
pub fn step_index(index: u8, triforce: u8, up: bool) -> u8 {
    let step = |i: u8| if up { i.wrapping_add(1) } else { i.wrapping_sub(1) };
    let mut i = step(index);
    if triforce == 0 {
        return i;
    }
    while triforce & (1 << (i & 7)) == 0 {
        i = step(i);
    }
    i
}

/// Facing bits ([`Direction::value`]) that step the index up (right, up) and down (left, down).
pub const FACING_UP: u8 = Direction::E.value() | Direction::N.value();
pub const FACING_DOWN: u8 = Direction::W.value() | Direction::S.value();

/// The fewest plays that leave the index on `level` (1-8) with a whirlwind out, and the facings ([`FACING_UP`],
/// [`FACING_DOWN`], or both) whose play starts such a sequence; (0, 0) when the index is there and a whirlwind is
/// already out (`summoned`). A whirlwind not yet summoned needs at least one play (the first makes it, and steps the
/// index). None without the level's piece.
pub fn plays(index: u8, triforce: u8, summoned: bool, level: u8) -> Option<(u32, u8)> {
    let target = level.checked_sub(1)? & 7;
    if triforce & (1 << target) == 0 {
        return None;
    }
    if summoned && index & 7 == target {
        return Some((0, 0));
    }
    // Breadth first over the index (mod 8), remembering the facing of each sequence's first play.
    let mut best: Option<(u32, u8)> = None;
    let mut frontier: Vec<(u8, u8)> = [(true, FACING_UP), (false, FACING_DOWN)]
        .into_iter()
        .map(|(up, facing)| (step_index(index, triforce, up) & 7, facing))
        .collect();
    for n in 1..=16u32 {
        for &(i, facing) in &frontier {
            if i == target {
                let (_, bits) = best.get_or_insert((n, 0));
                *bits |= facing;
            }
        }
        if best.is_some() {
            return best;
        }
        frontier = frontier
            .iter()
            .flat_map(|&(i, facing)| [true, false].map(|up| (step_index(i, triforce, up) & 7, facing)))
            .collect();
        frontier.sort_unstable();
        frontier.dedup();
    }
    None
}

/// Where the whirlwind of `state`'s room is (its position), if one is out.
pub fn whirlwind(state: &GameView) -> Option<(i32, i32)> {
    (1..=11u16)
        .find(|&slot| state.peek(OBJ_TYPE + slot) == OBJ_WHIRLWIND)
        .map(|slot| (state.peek(0x70 + slot) as i32, state.peek(0x84 + slot) as i32))
}

/// What the whirlwind wants of Link in `state` for it to set him down in the landing room of `level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Play the recorder `plays` more times, the next one facing one of `facings`.
    Play { plays: u32, facings: u8 },
    /// Stand still on the row of the whirlwind coming at him (`row`: its Y).
    Wait { row: i32 },
    /// Being carried off.
    Carried,
    /// The room's whirlwind was made and has gone (Link was hurt, held the clock or moved off its row): none again
    /// until the room is entered again (`IsWhirlwindSummoned`).
    Missed,
    /// Link lacks the level's piece: no whirlwind goes there.
    Never,
}

/// The [`Plan`] in `state` (an overworld room in play) for the landing room of `level`.
pub fn plan(state: &GameView, level: u8) -> Plan {
    if state.peek(TELEPORTING) != 0 {
        return Plan::Carried;
    }
    let summoned = state.peek(SUMMONED) != 0;
    let out = whirlwind(state);
    if summoned && out.is_none() {
        return Plan::Missed;
    }
    match plays(state.peek(LEVEL_INDEX), state.peek(INV_TRIFORCE), summoned, level) {
        None => Plan::Never,
        Some((0, _)) => Plan::Wait { row: out.map_or(state.link.position.y, |(_, y)| y) },
        Some((plays, facings)) => Plan::Play { plays, facings },
    }
}

/// What the engine plays of the route's whirlwind step in Link's room (`crate::Session::set_warp`, the step's level
/// from `crate::objectives::GameCompletion::warp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Warp {
    /// A dock warp (round 36, docs/routes/d3first.md "The wrong warps"): RECORDER in the water on the dock's row plays
    /// every play the index needs and waits for the pickup (`Executor::run_warp`). `unaided` (round 40, routes.yaml
    /// `macros: {unaided_docks: true}`): the mask offers the plays beside the family's own actions (the session's
    /// `unaided_dock_mask`); off, it leaves the play alone and gives way to a monster or a shot (`dock_warp_mask`).
    Dock { level: u8, unaided: bool },
    /// The whirlwind macro (round 40, routes.yaml `macros: {whirlwinds: [...]}`; decisions 2026-11-23 "The fresh
    /// generation's design calls"): RECORDER, offered wherever [`macro_facing`] finds the sequence playable from where
    /// Link stands, turns him to that facing, plays every play the index needs and waits for the pickup
    /// (`Executor::run_whirlwind`), whatever the monsters do.
    Whirlwind { level: u8 },
}

/// The facing the whirlwind macro ([`Warp::Whirlwind`]) plays its first play with in `state`, if the sequence can be
/// played from where Link stands: the plays are still to make ([`Plan::Play`]), the recorder can be played, and a
/// facing that starts the sequence is reached and its B press goes through (`crate::attack::plan`, the input model:
/// Link's position, facing and the border bands, never a monster or a shot). Link's own facing when it starts the
/// sequence (no turn), else the one of the fewest turn frames, horizontal first (a turn left or right keeps the
/// whirlwind's row and never walks into a cave mouth).
pub fn macro_facing(state: &GameView, level: u8) -> Option<Direction> {
    let Plan::Play { facings, .. } = plan(state, level) else { return None };
    if state.level() != 0 || !crate::actions::recorder_playable(state) {
        return None;
    }
    let facing = state.link.direction;
    if facings & facing.value() != 0 {
        return Some(facing);
    }
    [Direction::E, Direction::W, Direction::N, Direction::S]
        .into_iter()
        .filter(|d| facings & d.value() != 0)
        .filter_map(|d| crate::attack::plan(state, crate::ActionKind::Recorder, d).map(|frames| (frames, d)))
        .min_by_key(|&(frames, _)| frames)
        .map(|(_, d)| d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pieces 1-5 (the D5 triforce taken).
    const D5: u8 = 0b1_1111;

    /// late-mechanics.md §2.9's arithmetic, and Walks26's walk (round 26: from 0x0b with the index 0, one play facing
    /// down lands in 0x0b itself, two in 0x45; facing up one in 0x3c, two in 0x74).
    #[test]
    fn plays_step_past_missing_pieces() {
        assert_eq!(step_index(0, D5, false) & 7, 4, "0 -> 7 (L8 missing) -> 6 -> 5 -> 4 (L5)");
        assert_eq!(step_index(0, D5, false), 0xFC, "the raw index the game keeps");
        assert_eq!(step_index(0xFC, D5, false) & 7, 3, "L4: 0x45");
        assert_eq!(step_index(0, D5, true) & 7, 1, "L2: 0x3c");
        assert_eq!(step_index(1, D5, true) & 7, 2, "L3: 0x74");
        // After D6 (pieces 1-7, index 3) two plays down reach L2 (0x3c); after D8 (all, index 1) one reaches L1.
        assert_eq!(step_index(step_index(3, 0x7F, false), 0x7F, false) & 7, 1);
        assert_eq!(step_index(1, 0xFF, false) & 7, 0);
        assert_eq!(step_index(5, 0, true), 6, "no piece: one step, nothing appears");
    }

    #[test]
    fn plans_count_from_the_index() {
        // Out of Dungeon 5, index 0: two plays facing down (up takes three: 1, 2, 3).
        assert_eq!(plays(0, D5, false, 4), Some((2, FACING_DOWN)));
        // One play made (index 4, the whirlwind out): one more, down.
        assert_eq!(plays(0xFC, D5, true, 4), Some((1, FACING_DOWN)));
        // Both made: none, wait for it.
        assert_eq!(plays(0xFB, D5, true, 4), Some((0, 0)));
        // A stray overworld play facing up earlier (index 1): three down (0, 4, 3) or two up (2, 3).
        assert_eq!(plays(1, D5, false, 4), Some((2, FACING_UP)));
        // At the level already with no whirlwind out: away and back, either way first.
        assert_eq!(plays(3, D5, false, 4), Some((2, FACING_UP | FACING_DOWN)));
        // Without the level's piece: never.
        assert_eq!(plays(0, D5, false, 7), None);
    }
}
