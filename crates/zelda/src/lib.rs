//! Zelda game model and session, emulator-agnostic over [`Nes`].
//!
//! Ported from the old Python game stack (removed; history in docs/decisions.md 2026-11-05).

pub mod actions;
pub mod blocks;
pub mod bombs;
pub mod cheats;
pub mod change;
pub mod enums;
pub mod executor;
pub mod game;
pub mod game_map;
pub mod hearts;
pub mod ledger;
pub mod linger;
pub mod nes;
pub mod objectives;
pub mod power_on;
pub mod room;
pub mod room_entry;
pub mod room_state;
pub mod session;
pub mod tables;
pub mod tiles;
pub mod vars;
pub mod visits;

pub use actions::{Action, ActionMask, MASK_SIZE, compute_action_mask};
pub use change::{Change, LookAhead, Observed};
pub use enums::{ActionKind, Direction, MapLocation, Position, Tile};
pub use executor::{Interrupt, OptionResult, ZeldaError};
pub use game::GameView;
pub use game_map::GameMap;
pub use nes::Nes;
pub use session::{LoadOptions, Session, SessionCheckpoint, SessionSnapshot};
pub use vars::Var;
