//! Replays for watching (`play-zelda.py`).
//!
//! Nothing here plays the game live.  Simulations run headless and record every run as a replay
//! (`crates/replay`); the viewer only ever plays replays back:
//!
//! - [`queues`]: what is ready to watch.  Four first-in-first-out queues (`success`, `stuck`, `died`, and
//!   `random`, which takes a copy of any run) and `best`, the top runs seen (furthest along the route, then the
//!   most hearts, then the fewest frames).  [`Queues::wants_more`] says whether the simulations should run.
//! - [`feed`]: the background thread between the simulations (a [`Producer`]) and the queues.  It offers every
//!   finished run to the queues and pauses the simulations while nothing wants a run.
//! - [`screen`]: cells that play replays on the emulator alone, paced at a frame rate, each publishing its
//!   latest frame and Link's hearts.
//!
//! The crate knows replays and the emulator, not the game model, the envs or the models.

pub mod feed;
pub mod queues;
pub mod screen;

pub use feed::{Feed, Producer, Stats};
pub use queues::{Entry, QUEUES, Queue, Queues};
pub use screen::{CellState, Resolver, Screen, Status};

/// Link's health in half hearts and his heart containers, from `$66F` (containers - 1 in the high nibble,
/// filled hearts in the low one) and `$670` (the partial heart): `zelda::GameView::health` and `max_health`.
pub fn hearts(hearts_and_containers: u8, partial: u8) -> (u16, u8) {
    let containers = (hearts_and_containers >> 4) + 1;
    let filled = (hearts_and_containers & 0x0F).min(containers);
    let half = match partial {
        _ if filled == containers => 0,
        0 => 0,
        0x01..0x80 => 1,
        _ => 2,
    };
    ((u16::from(filled) * 2 + half).min(u16::from(containers) * 2), containers)
}

#[cfg(test)]
mod tests {
    use super::hearts;

    #[test]
    fn hearts_follow_the_games_two_bytes() {
        assert_eq!(hearts(0x22, 0xFF), (6, 3)); // three containers, all full
        assert_eq!(hearts(0x21, 0x7F), (3, 3)); // one full heart and a half
        assert_eq!(hearts(0x21, 0xFF), (4, 3)); // one full heart and a whole partial one
        assert_eq!(hearts(0x20, 0x00), (0, 3)); // dead
        assert_eq!(hearts(0x54, 0xFF), (10, 6));
        assert_eq!(hearts(0x2F, 0xFF), (6, 3)); // never more than the containers
    }
}
