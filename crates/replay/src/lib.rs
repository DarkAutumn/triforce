//! `TFREPLAY`: a recorded run that replays bit-exactly on the emulator alone (no env, reward or model
//! code): where it starts, every frame's buttons, every memory edit, the cheated-item ledger, the state it
//! ends in, how it ended, and optional streams (rewards, model decisions, keyframe states).  This crate
//! builds and parses the bytes; callers read and write the files.  Replays can be joined ([`stitch`]) and
//! cut at a keyframe ([`truncate`]) so a run continues from there.
//!
//! # File layout, version 0
//!
//! All integers are little-endian; `varint` is unsigned LEB128.
//!
//! | Offset | Field |
//! |---|---|
//! | 0 | magic `b"TFREPLAY"` |
//! | 8 | `version: u32` ([`REPLAY_VERSION`]) |
//! | 12 | `frames: u32` (input frames) |
//! | 16 | `ending: u8`: 0 unknown, 1 win, 2 death, 3 stuck, 4 other, 5 error |
//! | 17 | `source: u8`: 0 power on, 1 state file by name, 2 embedded state |
//! | 18 | `sections: u16` |
//! | 20 | ROM payload SHA-1, 20 bytes (bytes 10..30 of the END state blob) |
//! | 40 | reserved `u64` = 0 |
//! | 48 | section table: `sections` x 32 bytes `{kind: u16, 0u16, count: u32, last_frame: u32, 0u32, offset: u64, len: u64}`, ascending `kind`, each kind at most once |
//! | ... | section bytes, contiguous, in table order |
//!
//! A record's `frame` is the number of input frames run before it: a record at frame `k` applies after
//! input `k-1` and before input `k` (TFMOVIE's rule).  In every stream `dframe` is the varint difference
//! from the previous record's frame in that section (the first from 0); the table's `last_frame` is the
//! frame of the section's last record (0 when empty) and `count` its number of records.
//!
//! | Kind | Section | Records |
//! |---|---|---|
//! | 1 | META | one UTF-8 JSON object |
//! | 2 | SOURCE | source 0: `name_len: u8, name, delay: u32`; source 1: `file_hash: u64` ([`file_hash`]), `varint len, name`; source 2: empty |
//! | 3 | INPUTS | runs `(buttons: u8, varint count >= 1)`; `count` in the table is the number of runs; the run lengths sum to `frames` |
//! | 4 | EDITS | `(varint dframe, addr: u16, prev: u8, value: u8)` |
//! | 5 | CHEATS | `(varint dframe, item: u8 (0 bombs, 1 keys, 2 rupees), varint count)` |
//! | 6 | STATES | `(varint dframe, kind: u8, step: u16 (0xFFFF none), granted: [u8; 3] (bombs, keys, rupees), varint edits, len: u32, blob)`; kind 0 START, 1 END, 2 KEYFRAME, 3 CHECKPOINT; `edits` is the number of EDITS records that precede the state; `blob` is a `TFZSTATE` file |
//! | 7 | EVENTS | `(varint dframe, code: u8, payload)`: 1 CONTROL (no payload); 2 ROOM `(level: u8, room: u8, flags: u8 (bit 0 in cave), step: u16 (0xFFFF none), policy: u8 (0xFF none))`; 3 OUTCOME `(varint len, text)`; 4 EPISODE `(varint len, ending text, class: u8, flags: u8 (bit 0 terminated, bit 1 truncated))`; 5 SEGMENT (no payload) |
//! | 8 | REWARDS | `(varint dframe, term: u8, value: f32)`; `term` indexes META `reward_terms` |
//! | 9 | DECISIONS | `(varint dframe, policy: u8, kind: u8, direction: u8, duration: u8, value: f32, dist: [f32; N])`, `N` = META `actions.distribution` |
//!
//! META, SOURCE, INPUTS, EDITS, CHEATS, STATES and EVENTS are always present; STATES always has the END
//! (last, at frame `frames`) and, for an embedded source, the START (first, at frame 0).  REWARDS,
//! DECISIONS and KEYFRAME entries are optional.  A reader skips section kinds it does not know.
//!
//! # META
//!
//! The caller's object ([`RecordOptions::meta`]: `triforce`, `code`, `data`, `kind`, `route`, `stop_at`,
//! `models`, ...) plus, added by [`Recorder::finish`]: `seed`; `source` (`{"kind": "power-on", "name",
//! "delay"}`, `{"kind": "state", "name"}` or `{"kind": "checkpoint"}`); `ending` (`{"kind", "result",
//! "outcome", "reason"}`); `buttons` (the controller byte's bits); `cheat_items`; with REWARDS
//! `reward_terms` (names in first-use order); with DECISIONS `actions` ([`actions`]); after a stitch
//! `segments` (`{frames, seed, ending}` per joined replay).

mod edit;
mod format;
#[cfg(feature = "zcore")]
mod player;
mod reader;
mod recorder;

pub use edit::{stitch, truncate};
pub use format::{
    BUTTONS, CHEAT_ITEMS, Cheat, Context, Decision, ENDING_DEATH, ENDING_ERROR, ENDING_OTHER, ENDING_STUCK,
    ENDING_UNKNOWN, ENDING_WIN, ENDINGS, Edit, Ending, Event, EventRecord, MAGIC, OUTCOME_CLASSES,
    REPLAY_VERSION, ReplayError, Reward, STATE_KINDS, Source, State, StateKind, actions, file_hash, section,
};
#[cfg(feature = "zcore")]
pub use player::{Player, Report, verify};
pub use reader::{Records, Replay};
pub use recorder::{Mark, RecordOptions, Recorder};
