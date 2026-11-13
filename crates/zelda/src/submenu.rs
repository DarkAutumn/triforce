//! The item sub-screen: choosing the B item with the controller, as a player does (zelda-asm `mechanics.md` "Play
//! update, pause and submenu"; asm:Submenu_UpdateActive, asm:Submenu_UpdateSelection,
//! asm:Inv_FindAndSelectOccupiedSlot).
//!
//! START opens it (asm:Mode5Play_Update@NotInMenu); it scrolls in (64 frames in a dungeon, 65 in the overworld from
//! the press to the first active frame), moves the selection one occupied slot per change of the held left/right
//! input (a diagonal counts as a change, so alternating Right and Right+Up steps every frame), and START in the
//! active state closes it after that frame's selection (59 frames of scrolling). The game is paused throughout: only
//! the random numbers, the frame counter and the sound run (asm:Nmi_Isr@UpdateTimers).

use crate::enums::ActionKind;
use crate::game::GameView;

pub const ADDR_SELECTED_ITEM: usize = 0x656;
pub const ADDR_SUBMENU_STATE: usize = 0xE1;
pub const ADDR_SEARCH_DIR: usize = 0xEF;
const ADDR_FLUTE_TIMER: u16 = 0x3C;
const ADDR_PAUSE_STATE: u16 = 0xE0;
const ADDR_ROOM_BRIGHTENING: u16 = 0x51E;
const ADDR_MODE: u16 = 0x12;
/// asm:InvItems: item slot n is `INV_ITEMS + n` ($657 the sword, $658 bombs, $659 arrows, $65A bow, ...).
const INV_ITEMS: u16 = 0x657;
const WOODEN_BOOMERANG: u16 = INV_ITEMS + 0x1D;
const MAGIC_BOOMERANG: u16 = INV_ITEMS + 0x1E;
const BOW: u16 = INV_ITEMS + 3;
const POTION: u16 = INV_ITEMS + 7;
const LETTER: u16 = INV_ITEMS + 0x0F;
const LETTER_SLOT: u8 = 0x0F;

/// The game's direction bits as `SubmenuSearchDir` stores them.
pub const SEARCH_RIGHT: u8 = 1;
pub const SEARCH_LEFT: u8 = 2;
pub const SEARCH_UP: u8 = 8;

/// Frame caps of the selection (never reached in play: the opening takes 64-65 frames, the closing 59, at most 4
/// steps in the 9-slot ring).
pub const MAX_OPEN_FRAMES: u32 = 80;
pub const MAX_SELECT_FRAMES: u32 = 12;
pub const MAX_CLOSE_FRAMES: u32 = 80;

/// The B item slot an attack uses (`SelectedItemSlot`: 0 boomerang, 1 bombs, 2 arrow).
pub fn slot(kind: ActionKind) -> Option<u8> {
    match kind {
        ActionKind::Boomerang => Some(0),
        ActionKind::Bomb => Some(1),
        ActionKind::Arrow => Some(2),
        _ => None,
    }
}

/// `kind` is a B item that is not the selected one: the action opens the sub-screen first.
pub fn needs_selection(view: &GameView, kind: ActionKind) -> bool {
    slot(kind).is_some_and(|s| view.peek(ADDR_SELECTED_ITEM as u16) != s)
}

/// START opens the sub-screen now: a play mode (5, 9, $B, $C), not paused (Select or a potion's refill), no flute
/// tune and no candle brightening the room (asm:Mode5Play_Update), the sub-screen closed.
pub fn can_open(view: &GameView) -> bool {
    matches!(view.peek(ADDR_MODE), 5 | 9 | 0x0B | 0x0C)
        && view.peek(ADDR_FLUTE_TIMER) == 0
        && view.peek(ADDR_ROOM_BRIGHTENING) == 0
        && view.peek(ADDR_PAUSE_STATE) == 0
        && view.peek(ADDR_SUBMENU_STATE as u16) == 0
}

/// `SubmenuState` of the active sub-screen: 7 in a dungeon, 8 in the overworld (one more set-up state).
pub fn active_state(level: u8) -> u8 {
    if level == 0 { 8 } else { 7 }
}

/// asm:Inv_StepItemSlotInSearchDir: one place around the ring 0-8.
fn step_slot(slot: u8, dir: u8) -> u8 {
    match dir & 3 {
        0 => slot,
        d if d & 1 != 0 => (slot + 1) % 9,
        _ => (slot + 8) % 9,
    }
}

/// asm:Inv_FindAndSelectOccupiedSlot with a stepping `dir` (right or left): the slot the selection moves to from
/// `from` (`peek` reads RAM by CPU address). Slot 0 counts with either boomerang, slot 3 (bow) never, the arrow
/// only with the bow, the letter stands in for an empty potion slot; ten tries, then slot 0.
pub fn search(peek: impl Fn(u16) -> u8, from: u8, dir: u8) -> u8 {
    let mut slot = if from == LETTER_SLOT { 7 } else { from };
    let mut tries = 10;
    // The arrow-without-bow path retries without counting; it ends because every call here steps.
    for _ in 0..64 {
        slot = step_slot(slot, dir);
        let found = match slot {
            0 => (peek(WOODEN_BOOMERANG) != 0 || peek(MAGIC_BOOMERANG) != 0).then_some(0),
            3 => None,
            s if peek(INV_ITEMS + s as u16) != 0 => {
                if s == 2 && peek(BOW) == 0 {
                    continue;
                }
                Some(s)
            }
            7 if peek(LETTER) != 0 && peek(POTION) == 0 => Some(LETTER_SLOT),
            _ => None,
        };
        if let Some(found) = found {
            return found;
        }
        tries -= 1;
        if tries == 0 {
            break;
        }
    }
    0
}

/// Steps from `from` to `to` going `dir`, if the search gets there within a ring's length.
pub fn steps(peek: impl Fn(u16) -> u8 + Copy, from: u8, to: u8, dir: u8) -> Option<u32> {
    let mut slot = from;
    for n in 1..=9 {
        slot = search(peek, slot, dir);
        if slot == to {
            return Some(n);
        }
    }
    None
}

/// The way to the slot: right or left, whichever takes fewer steps (right on a tie), and the step count.
pub fn way(peek: impl Fn(u16) -> u8 + Copy, from: u8, to: u8) -> Option<(u8, u32)> {
    let right = steps(peek, from, to, SEARCH_RIGHT).map(|n| (SEARCH_RIGHT, n));
    let left = steps(peek, from, to, SEARCH_LEFT).map(|n| (SEARCH_LEFT, n));
    match (right, left) {
        (Some(r), Some(l)) => Some(if l.1 < r.1 { l } else { r }),
        (r, l) => r.or(l),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory(items: &[(u16, u8)]) -> impl Fn(u16) -> u8 + Copy {
        let mut ram = [0u8; 0x800];
        for &(addr, v) in items {
            ram[addr as usize] = v;
        }
        move |a: u16| ram[a as usize]
    }

    #[test]
    fn the_search_skips_empty_slots_and_the_bow() {
        // Boomerang, bombs, bow and arrows: 0 -> 1 -> 2 -> 0 to the right.
        let inv = inventory(&[(WOODEN_BOOMERANG, 1), (INV_ITEMS + 1, 4), (INV_ITEMS + 2, 1), (BOW, 1)]);
        assert_eq!(search(inv, 0, SEARCH_RIGHT), 1);
        assert_eq!(search(inv, 1, SEARCH_RIGHT), 2);
        assert_eq!(search(inv, 2, SEARCH_RIGHT), 0);
        assert_eq!(search(inv, 0, SEARCH_LEFT), 2);
        assert_eq!(way(inv, 0, 2), Some((SEARCH_LEFT, 1)));
        assert_eq!(way(inv, 2, 1), Some((SEARCH_LEFT, 1)));
        // Arrows without the bow are skipped.
        let no_bow = inventory(&[(WOODEN_BOOMERANG, 1), (INV_ITEMS + 1, 4), (INV_ITEMS + 2, 1)]);
        assert_eq!(search(no_bow, 1, SEARCH_RIGHT), 0);
        assert_eq!(steps(no_bow, 0, 2, SEARCH_RIGHT), None);
    }
}
