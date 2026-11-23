//! `TFREPLAY`: a recorded run that replays bit-exactly on the emulator alone (no env, reward or model
//! code): where it starts, every frame's buttons, every memory edit, the cheated-item ledger, the state it
//! ends in, how it ended, what the agent was steered to and could do (TARGETS, MASKS), what it thought
//! (DECISIONS), and optional streams (rewards, keyframe states).  This crate builds and parses the bytes; callers
//! read and write the files.  Replays can be joined ([`stitch`]) and cut at a keyframe ([`truncate`]) so a run
//! continues from there.
//!
//! # Versions
//!
//! Every format change bumps [`REPLAY_VERSION`] (the owner, Nov 23: replays never need to be readable by older
//! readers).  A reader refuses a newer replay ("replay version N; this reader reads up to M: update") and reads
//! every older one.  Version 0: the first; 1 (round 32): STATES' route grants; 2 (Nov 23): TARGETS, MASKS, the
//! quantized DECISIONS with the draw, the danger and the foresight, the VISIT event, META `tables`, DECISIONS
//! always recorded.
//!
//! # Self-description
//!
//! A replay names everything it records in its own META, so a reader holding only this format (no route data,
//! network or code of the version that recorded it) can print, for any frame, which network acted, the legal
//! actions, the chosen one, the probabilities, the danger per action and the objective with its targets, each
//! labelled: META `actions` describes the action space and the distribution's layout, `models[i]` the network
//! of policy index `i` (family, run, checkpoint and its SHA-1, architecture and parameters, act mode, and its
//! `outputs`: name, shape, labels, meaning and units of each output, the hit head's among them where it has one),
//! and `tables` the names TARGETS, MASKS and DECISIONS index.  A new output, role or table entry is written with
//! its META description.
//!
//! # File layout, version 2
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
//! | 6 | STATES | `(varint dframe, kind: u8, step: u16 (0xFFFF none), granted: [u8; 3] (bombs, keys, rupees: a room's grants), route_granted: [u8; 3] (a route step's grants; version 1, absent in version 0), varint edits, len: u32, blob)`; kind 0 START, 1 END, 2 KEYFRAME, 3 CHECKPOINT; `edits` is the number of EDITS records that precede the state; `blob` is a `TFZSTATE` file |
//! | 7 | EVENTS | `(varint dframe, code: u8, payload)`: 1 CONTROL (no payload); 2 ROOM `(level: u8, room: u8, flags: u8 (bit 0 in cave), step: u16 (0xFFFF none), policy: u8 (0xFF none))`; 3 OUTCOME `(varint len, text)`; 4 EPISODE `(varint len, ending text, class: u8, flags: u8 (bit 0 terminated, bit 1 truncated))`; 5 SEGMENT (no payload); 6 VISIT `(varint len, JSON object)`: a room visit's record as the env judged it, written when it closes (version 2) |
//! | 8 | REWARDS | `(varint dframe, term: u8, value: f32)`; `term` indexes META `reward_terms` |
//! | 9 | DECISIONS | version 2: `(varint dframe, policy: u8, kind: u8, direction: u8, duration: u8, draw: u8 (bits 0-6 index META tables.draws, bit 7 the act mode's low-health part), value: f32, flags: u8 (bit 0 danger, bit 1 foresight), dist: [u8; N] (255 = -inf, masked; else -byte/16 nats), [danger: varint n, [u8; n]: P(hit) x 255 per legal joint action, joint order], [foresight: outcome: u8 (1 safe, 2 replaced, 3 unsafe), varint trials, varint frames, n: u8, [u16; n] vetoed joint actions in the order tried])`; versions 0-1: `(varint dframe, policy: u8, kind: u8, direction: u8, duration: u8, value: f32, dist: [f32; N])`; `N` = META `actions.distribution`, a joint action is legal when its kind's, direction's and duration's entries are finite ([`legal_joint`]) |
//! | 10 | TARGETS | `(varint dframe, objective: u8 (META tables.objective_kinds), flags: u8 (bit 0 treasure refused, 1 drops first, 2 bag drops, 3 the path search knows the clips), route: u8 (tables.routes, 0xFF none), step: u16 (0xFFFF none), way: u8 (0 none, 1 clips, 2 plain), stretch: u8 (tables.stretches, 0xFF none), n: u8, n x (level, room, in_cave: u8) next rooms, n: u8, n x the same retreat rooms, n: u8, n x (role: u8 (tables.target_roles), flags: u8 (bit 0 obs, bit 1 reward-only), fields: [u8; 4] (the role's named fields)))`, written at a decision when it differs from the last (version 2) |
//! | 11 | MASKS | `(varint dframe, kinds: u16 (bit k: META actions.kinds[k] legal), per legal kind in order (directions: u8, durations: u8 bits), n: u8, n x (kind: u8, reason: u8 (tables.mask_reasons)))`, written at a decision when it differs from the last (version 2) |
//!
//! META, SOURCE, INPUTS, EDITS, CHEATS, STATES and EVENTS are always present; STATES always has the END
//! (last, at frame `frames`) and, for an embedded source, the START (first, at frame 0).  From version 2
//! DECISIONS, TARGETS and MASKS are always written; REWARDS and KEYFRAME entries are optional.  The provenance
//! label of TARGETS (route, step, way, stretch) is for people; no metric may depend on it.
//!
//! # META
//!
//! The caller's object ([`RecordOptions::meta`]: `triforce`, `code`, `data`, `kind`, `route`, `stop_at`,
//! `models`, ...) plus, added by [`Recorder::finish`]: `seed`; `source` (`{"kind": "power-on", "name",
//! "delay"}`, `{"kind": "state", "name"}` or `{"kind": "checkpoint"}`); `ending` (`{"kind", "result",
//! "outcome", "reason"}`); `rng_phase` when the start wrote a natural RNG state (its steps from reset; the
//! bytes are the frame-0 EDITS of `$18-$24`); `buttons` (the controller byte's bits); `cheat_items`; with
//! REWARDS `reward_terms` (names in first-use order); `actions` ([`actions`]); the env's entries
//! ([`Recorder::set_meta`]: `tables`, a chain run's `run`); after a stitch `segments` (`{frames, seed, ending}` per
//! joined replay, with its `rng_phase` when it has one).

mod edit;
mod format;
mod names;
#[cfg(feature = "zcore")]
mod player;
mod reader;
mod recorder;

pub use edit::{stitch, truncate};
pub use format::{
    BUTTONS, CHEAT_ITEMS, Cheat, Context, Decision, Dist, ENDING_DEATH, ENDING_ERROR, ENDING_OTHER, ENDING_STUCK,
    ENDING_UNKNOWN, ENDING_WIN, ENDINGS, Edit, Ending, Event, EventRecord, Foresight, MAGIC, Mask,
    OBJECTIVE_BAG_DROPS, OBJECTIVE_CLIPS, OBJECTIVE_DROPS_FIRST, OBJECTIVE_TREASURE_REFUSED, OUTCOME_CLASSES,
    REPLAY_VERSION, ReplayError, Reward, RoomRef, STATE_KINDS, Source, State, StateKind, TARGET_OBS,
    TARGET_REWARD_ONLY, TargetRef, Targets, actions, file_hash, legal_joint, log_prob_of, quantize_log_prob,
    quantize_probability, section,
};
pub use names::{Names, room_text};
#[cfg(feature = "zcore")]
pub use player::{Player, Report, verify};
pub use reader::{Records, Replay};
pub use recorder::{Mark, RecordOptions, Recorder, Thought};
