//! Zelda game model and session, emulator-agnostic over [`Nes`].
//!
//! A mechanical port of the Python game stack (`triforce/`): see each module's docs for its Python source.

pub mod actions;
pub mod blocks;
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
pub mod room;
pub mod room_entry;
pub mod room_state;
pub mod session;
pub mod tables;
pub mod tiles;
pub mod vars;
pub mod visits;

pub use actions::{Action, ActionMask, MASK_SIZE, compute_action_mask};
pub use change::Change;
pub use enums::{ActionKind, Direction, MapLocation, Position, Tile};
pub use executor::{Interrupt, OptionResult, ZeldaError};
pub use game::GameView;
pub use game_map::GameMap;
pub use nes::Nes;
pub use session::{LoadOptions, Session, SessionSnapshot};
pub use vars::Var;
