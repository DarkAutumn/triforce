//! The grid-point skip as an option (round 21; `zelda-asm/mechanics.md`, "# Screen wrap (overworld)" and
//! "# Grid-point skip: block clip, ladder clip and their run-up").
//!
//! One frame of perpendicular input 5 pixels short of a grid point T, in the right sub-pixel cadence, lets Link cross
//! T with no frame starting on it, so T's tile and screen-edge tests are skipped. The agent's MOVE (4-32 frames)
//! can't do that, so the executor plays it as a macro ([`Controller`], driven by `crate::executor`): it reads
//! `$70`/`$84`, `$394`, `$3A8` and `$64` every frame and follows the decision table of mechanics.md (back off to G'
//! when too close or in the wrong cadence, walk to the trick state, press one perpendicular frame, hold the heading).
//! No RAM is written.
//!
//! Four uses: the overworld **wrap** (W, E): T is the screen bound, Link comes in at the opposite side of the same
//! room and the macro walks on to where that side's ordinary room change is one MOVE away (X `$F0` after a W wrap,
//! `$00` after an E wrap); the overworld **wrap exit** (E, round 22): an E wrap whose landing at X `$00` is solid,
//! offered only on the rows of a game.yaml W `wrap` exit; Left on the landing frame fires the W screen edge test
//! whatever the tile, and the macro ends in the room to the W (0x75 into 0x74, mechanics.md "After the wrap, edge
//! by edge"); the dungeon **block clip** (N, S): T is where a 16-pixel solid square stops Link, the skip carries him
//! 8 pixels into it (T -+ 8), from where a sideways MOVE goes on; and the dungeon **ladder clip** (any heading, Link
//! holds the ladder): T is the laid ladder's distance-0 point, where water two or three squares wide stops the plain
//! ladder; the skip carries Link over all of it (the relay lays a second ladder where needed) onto the land beyond.
//!
//! [`skip_target`] (state only) is the single availability rule of the mask, the executor and the observation;
//! [`skip_jumps`] is the same rule from Link's rest positions, the jumps the path search adds
//! (`crate::room::Room::wavefront`). A specialist's `clips: true` turns both on: the CLIP kind ([`allow_clips`]) and
//! the jumps. MOVE never performs a skip.

use crate::actions::{ActionMask, CARDINALS};
use crate::change::Change;
use crate::enums::{ActionKind, Direction, GAMEPLAY_START_Y, Position};
use crate::executor::Interrupt;
use crate::game::{GameView, at_dungeon_bound};
use crate::room::{Jump, OW_WATER_TILES, Room, WATER_TILE};
use crate::tables::{OBJ_DIRECTION, OBJ_ID, OBJ_POS_X, OBJ_POS_Y, OBJ_STATUS};
use crate::vars::Var;
use std::ops::AddAssign;

/// Overworld screen bounds on the X axis (asm:Link_ScreenEdgeBounds).
const OW_BOUND_W: u8 = 0x00;
const OW_BOUND_E: u8 = 0xF0;
/// The wrap zone: at most 32 pixels from the bound (the bound itself included).
const WRAP_ZONE: u8 = 0x20;
/// The overworld slow tiles (asm:Link_InitSpeed): they change Link's speed and zero `ObjPosFrac`, which the
/// decision table does not model.
const SLOW_TILES: [u8; 2] = [0x74, 0x75];
/// Dungeon room bounds Link can stand on (asm:Room_ObjBoundsUW: X $20..$D0, Y $5D..$BD).
const UW_LEFT: i32 = 0x20;
const UW_RIGHT: i32 = 0xD0;
const UW_TOP: i32 = 0x5D;
const UW_BOTTOM: i32 = 0xBD;
/// Overworld screen bounds Link can stand on (asm:Link_ScreenEdgeBounds: X $00..$F0, Y $3D..$DD).
const OW_LEFT: i32 = OW_BOUND_W as i32;
const OW_RIGHT: i32 = OW_BOUND_E as i32;
const OW_TOP: i32 = 0x3D;
const OW_BOTTOM: i32 = 0xDD;
/// Dungeon outer border band (asm:Link_BorderBounds): a perpendicular press is dropped there, on the facing's axis
/// (left X < $17, right X >= $D9, up Y < $55, down Y >= $C6).
const UW_BAND_W: i32 = 0x17;
const UW_BAND_E: i32 = 0xD9;
const UW_BAND_N: i32 = 0x55;
const UW_BAND_S: i32 = 0xC6;
/// Play mode.
const MODE_PLAY: u8 = 5;

/// `ObjPosFrac` of Link (`$3A8`): the sub-pixel that sets the 1-2 pixel cadence.
const LINK_POS_FRAC: usize = Var::LinkPosFrac.addr() as usize;
const LINK_X: usize = Var::LinkX.addr() as usize;
const LINK_Y: usize = Var::LinkY.addr() as usize;
const LINK_GRID_OFFSET: usize = Var::LinkGridOffset.addr() as usize;
const LINK_DIRECTION: usize = Var::LinkDirection.addr() as usize;
/// `LadderSlot` (`$64`): the laid ladder's object slot, 0 = none out (asm:Ladder_CheckAndDraw).
const LADDER_SLOT: usize = Var::LadderSlot.addr() as usize;
/// The highest monster slot (asm:Obj_FindEmptyMonsterSlot searches $0B down to $01).
const LAST_MONSTER_SLOT: usize = 0x0B;

/// Frames the macro may take before it gives up (a run-up from 32 pixels with two back-offs takes about 90; the
/// longest ladder clip with a back-off about 60).
pub const MAX_SKIP_FRAMES: u32 = 200;
/// Presses the macro may make before it gives up (each failed one costs a back-off).
pub const MAX_SKIP_PRESSES: u8 = 3;
/// Frames in a row Link may stand still (no lag frame repeats this often; a ladder put away costs one) before the
/// macro gives up.
pub const MAX_STILL_FRAMES: u32 = 4;

/// What the skip does in the direction it is offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkipKind {
    /// Overworld screen wrap (W, E).
    Wrap,
    /// Overworld wrap exit (E): the E wrap onto a solid landing, then Left out of the W edge (game.yaml `wrap`).
    WrapExit,
    /// Dungeon block clip (N, S).
    Block,
    /// Dungeon ladder clip (any heading).
    Ladder,
}

impl SkipKind {
    pub const fn name(self) -> &'static str {
        match self {
            SkipKind::Wrap => "wrap",
            SkipKind::WrapExit => "wrap-exit",
            SkipKind::Block => "block",
            SkipKind::Ladder => "ladder",
        }
    }
}

/// Where the skip goes: the heading, T on the heading's axis and where the macro ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkipTarget {
    pub direction: Direction,
    pub kind: SkipKind,
    /// The grid point whose tests are skipped (X for W/E, Y for N/S).
    pub t: u8,
    /// Distance still to go to T in the heading's sense when the macro is done (negative: past T). A wrap ends 16
    /// pixels past the bound (X `$F0` after a W wrap, `$00` after an E wrap; a wrap exit presses Left there and ends
    /// in the next room); a block clip 8 pixels into the square; a ladder clip on the land beyond the water with no
    /// ladder out (`$20`-`$30` past T, [`ladder_reach`]).
    pub end: i8,
}

impl SkipTarget {
    /// Heading toward smaller coordinates (W, N).
    fn negative(&self) -> bool {
        matches!(self.direction, Direction::W | Direction::N)
    }

    fn horizontal(&self) -> bool {
        matches!(self.direction, Direction::W | Direction::E)
    }

    /// Link's state at the start of a frame, in the heading's sense (see [`SkipState`]).
    pub fn read(&self, ram: &[u8]) -> SkipState {
        let c = if self.horizontal() { ram[LINK_X] } else { ram[LINK_Y] };
        let o = ram[LINK_GRID_OFFSET] as i8;
        let f = ram[LINK_POS_FRAC];
        let (p, o_h, good) = if self.negative() {
            (c.wrapping_sub(self.t) as i8, o.wrapping_neg(), f == 0x80)
        } else {
            (self.t.wrapping_sub(c) as i8, o, f == 0x00)
        };
        let facing = ram[LINK_DIRECTION] == self.direction.value();
        // A ladder Link has been on (object state 2), not one just laid ahead of him (state 1).
        let slot = ram[LADDER_SLOT] as usize;
        let ladder = (1..=LAST_MONSTER_SLOT).contains(&slot) && ram[OBJ_STATUS.0 + slot] == 2;
        SkipState { p, o: o_h, good, facing, ladder, raw: (c, ram[LINK_GRID_OFFSET], f, ram[LINK_DIRECTION]) }
    }

    /// The controller byte that walks Link along the heading (`toward`) or away from it.
    pub fn walk(&self, toward: bool) -> u8 {
        let d = if toward { self.direction } else { self.direction.opposite() };
        crate::executor::direction_buttons(d)
    }

    /// The one perpendicular frame: Up for a W/E heading (the heading released, mechanics.md "The rule"), Right for
    /// N/S.
    pub fn press(&self) -> u8 {
        if self.horizontal() {
            crate::executor::buttons::UP
        } else {
            crate::executor::buttons::RIGHT
        }
    }
}

/// Link on the heading's axis at the start of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkipState {
    /// Distance to T in the heading's sense (8-bit, signed: negative once past T).
    pub p: i8,
    /// `ObjGridOffset` in the heading's sense: positive = moved toward T since the last grid point.
    pub o: i8,
    /// `ObjPosFrac` is the good fraction for the heading (`$80` W/N, `$00` E/S): the next frame toward T moves 1 pixel.
    pub good: bool,
    /// `ObjDir` is the heading (case 6 needs the offset's sign to agree with it: a Link walking away from T at the
    /// trick position only turns back).
    pub facing: bool,
    /// A ladder Link has been on is out (`LadderSlot` != 0, its state 2): a ladder clip is done only once the last
    /// one is put away. One just laid ahead of him at the end (two squares crossed, the walk on arrives facing more
    /// water) stays, as after any walk toward water.
    pub ladder: bool,
    /// (coordinate, `$394`, `$3A8`, `$98`) as read.
    pub raw: (u8, u8, u8, u8),
}

impl SkipState {
    /// The trick state: 5 pixels short of T, offset 3 into the cell, good fraction, heading toward T.
    pub fn is_trick(&self) -> bool {
        self.p == 5 && self.o == 3 && self.good && self.facing
    }
}

/// The macro's phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Walking toward the trick state.
    Approach,
    /// Walking away until on G' (16 pixels before T) with offset 0.
    BackOff,
    /// Pressed: holding the heading until the end point.
    Crossing,
    /// A wrap exit at its end point (X `$00`, inside the solid left edge): holding Left until the room changes.
    Exit,
}

/// One frame's input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Toward,
    Away,
    Press,
    /// Nothing held: at a ladder clip's end it puts the last ladder away and lays no new one.
    Release,
}

/// The per-frame decision table (mechanics.md "Decision table"), first matching row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Controller {
    pub target: SkipTarget,
    pub phase: Phase,
    pub presses: u8,
}

/// What the controller wants next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Input(Input),
    Done,
    /// The crossing ended on T (the skip was missed) more often than [`MAX_SKIP_PRESSES`] allows, or the presses ran
    /// out.
    GiveUp,
}

impl Controller {
    pub fn new(target: SkipTarget) -> Controller {
        Controller { target, phase: Phase::Approach, presses: 0 }
    }

    /// The input for the frame that starts in `s`.
    pub fn decide(&mut self, s: SkipState) -> Decision {
        match self.phase {
            Phase::Exit => Decision::Input(Input::Away),
            Phase::Crossing => {
                // At the end point. A ladder out there is put away by one frame with nothing held (a held heading
                // would lay another over water ahead: the relay; asm:Ladder_CheckAndDraw@StashLadder, state 2).
                if s.p == self.target.end && s.o == 0 {
                    if self.target.kind == SkipKind::Ladder && s.ladder {
                        return Decision::Input(Input::Release);
                    }
                    // A wrap exit turns back on the landing frame: on the bound, Left fires the W edge test whatever
                    // the tile (asm:Link_StopAtUnwalkableTile); a held Right would walk on into the room (column 2).
                    if self.target.kind == SkipKind::WrapExit {
                        self.phase = Phase::Exit;
                        return Decision::Input(Input::Away);
                    }
                    return Decision::Done;
                }
                if s.p == 0 && s.o == 0 {
                    // Stopped on T: the press was missed (the edge, the square's or the ladder's test was made).
                    // Never hold the heading here (at a screen bound that is the ordinary room change): back off
                    // and try again.
                    self.phase = Phase::BackOff;
                    return self.decide(s);
                }
                if s.p < 0 || (s.p > 0 && s.o < 0) {
                    return Decision::Input(Input::Toward);
                }
                // Toward T with the offset counting away from it: not the crossing a press starts.
                self.phase = Phase::BackOff;
                self.decide(s)
            }
            Phase::BackOff => {
                if s.p > 16 || (s.p == 16 && s.o == 0) {
                    self.phase = Phase::Approach;
                    return self.decide(s);
                }
                Decision::Input(Input::Away)
            }
            Phase::Approach => {
                if s.is_trick() {
                    if self.presses >= MAX_SKIP_PRESSES {
                        return Decision::GiveUp;
                    }
                    self.presses += 1;
                    return Decision::Input(Input::Press);
                }
                // One frame before the trick state (G - 1 toward T with the other fraction).
                if s.p == 7 && s.o == 1 && !s.good {
                    return Decision::Input(Input::Toward);
                }
                // On G with the good fraction: 2 frames, then the trick state.
                if s.p == 8 && s.o == 0 && s.good {
                    return Decision::Input(Input::Toward);
                }
                // Between G' and G (either way: walking on retraces onto G with the fraction he had there), on G'
                // or farther back.
                if s.p > 8 {
                    return Decision::Input(Input::Toward);
                }
                // On G with the other fraction, on T, or anywhere between G and T but the two states above.
                self.phase = Phase::BackOff;
                self.decide(s)
            }
        }
    }

    /// After the press frame: the press took (offset rewritten from T's side, Link stepped back) or not (a lag frame
    /// repeats the trick state, anything else is handled by the table).
    pub fn pressed(&mut self, s: SkipState) {
        if s.p == 7 && s.o == -7 {
            self.phase = Phase::Crossing;
        }
    }
}

/// Where the skip in `direction` is available in `state`, and where it goes (mechanics.md; round 21 contract
/// "Availability"). Overworld W/E: the wrap, E also the wrap exit, else (round 24) the ladder clip in a room where the
/// game lays the ladder (0x5f's heart container from the wrap arrival on its water); dungeon: the ladder clip, else
/// (N/S) the block clip when it is useful. None elsewhere, and none while a Like-Like holds Link: the game discards
/// every direction then (asm:Link_Update; Noops52, Nov 27: in 9:0x51 CLIP N was pressed 33 times in a hold, to death).
pub fn skip_target(state: &GameView, direction: Direction) -> Option<SkipTarget> {
    if state.mode() != MODE_PLAY || state.link.status != 0 || state.var(Var::LinkShoveDir) != 0 || state.link_held() {
        return None;
    }
    let ladder_out = state.var(Var::LadderSlot) != 0;
    match (state.level(), direction) {
        (0, Direction::W | Direction::E) => {
            let wrap = if ladder_out { None } else { wrap_target(state, direction) };
            wrap.or_else(|| ladder_target(state, direction))
        }
        (0, _) => None,
        (_, Direction::N | Direction::S) => ladder_target(state, direction).or_else(|| {
            if ladder_out { None } else { block_target(state, direction) }
        }),
        _ => ladder_target(state, direction),
    }
}

pub fn skip_available(state: &GameView, direction: Direction) -> bool {
    skip_target(state, direction).is_some()
}

/// Allows CLIP in every direction a skip is available but `except` (the session does it while a specialist with
/// `clips: true` has the run-time switch on; `except` are the wraps the objective vetoes,
/// `crate::objectives::wraps_into_wrong_exits`).
pub fn allow_clips(mask: &mut ActionMask, state: &GameView, except: &[Direction]) {
    for d in CARDINALS {
        if !except.contains(&d) && skip_available(state, d) {
            mask.allow(ActionKind::Clip, Some(d));
        }
    }
}

fn wrap_target(state: &GameView, direction: Direction) -> Option<SkipTarget> {
    let offset = state.var(Var::LinkGridOffset);
    let facing = state.var(Var::LinkDirection) as u8;
    // The offset (if any) along the row.
    if offset != 0 && !matches!(facing, 1 | 2) {
        return None;
    }
    let kind = wrap_at(state.room(), state.link.position.x, state.link.position.y, direction)?;
    let t = if direction == Direction::W { OW_BOUND_W } else { OW_BOUND_E };
    Some(SkipTarget { direction, kind, t, end: -16 })
}

/// The wrap heading `direction` (W or E) for Link at (x, y) in the overworld `room`, or None (position only;
/// [`wrap_target`] adds Link's state): on a row line in the zone, every tile the macro walks over on the row (G' to the
/// bound, the back-off included) open, and the landing cells at the opposite edge open (mechanics.md "After the wrap,
/// edge by edge"; a solid one strands Link). An E wrap onto solid landing cells is the wrap exit where the room's
/// game.yaml W `wrap` exit lists Link's tile row (`Room::wrap_exit`): there the arrival in the room to the W lets him
/// walk on; elsewhere it is a softlock or no way out at all.
fn wrap_at(room: &Room, x: i32, y: i32, direction: Direction) -> Option<SkipKind> {
    if y & 7 != 5 || !(0x3D..=0xDD).contains(&y) {
        return None;
    }
    let (zone, run_up, landing): (bool, [i32; 4], [i32; 2]) = match direction {
        Direction::W => (x <= WRAP_ZONE as i32, [0, 1, 2, 3], [0x1E, 0x1F]),
        _ => ((0xD0..=OW_BOUND_E as i32).contains(&x), [0x1C, 0x1D, 0x1E, 0x1F], [0, 1]),
    };
    let row = (y - 53).div_euclid(8);
    let open = |c: i32| room.is_tile_walkable(c, row) && !SLOW_TILES.contains(&room.tile(c, row));
    if !zone || !run_up.into_iter().all(open) {
        return None;
    }
    if landing.into_iter().all(open) {
        return Some(SkipKind::Wrap);
    }
    let tile_row = Position::new(x, y).tile().1;
    (direction == Direction::E && room.wrap_exit().is_some_and(|(lo, hi)| (lo..=hi).contains(&tile_row)))
        .then_some(SkipKind::WrapExit)
}

fn block_target(state: &GameView, direction: Direction) -> Option<SkipTarget> {
    let (x, y) = (state.link.position.x, state.link.position.y);
    let offset = state.var(Var::LinkGridOffset);
    let facing = state.var(Var::LinkDirection) as u8;
    if state.var(Var::DoorwayDir) != 0 || (offset != 0 && !matches!(facing, 4 | 8)) {
        return None;
    }
    let t = block_at(state.room(), x, y, direction)?;
    Some(SkipTarget { direction, kind: SkipKind::Block, t: t as u8, end: -8 })
}

/// T of the block clip heading `direction` (N or S) for Link at (x, y), or None (position only; [`block_target`]
/// adds Link's state).
fn block_at(room: &Room, x: i32, y: i32, direction: Direction) -> Option<i32> {
    if x & 7 != 0 || !(UW_BAND_W..UW_BAND_E).contains(&x) {
        return None;
    }
    let step: i32 = if direction == Direction::N { -8 } else { 8 };
    let back = direction.opposite();
    let blocked = |g: i32| !room.can_link_move_from(x, g, direction);
    // The first grid point at or ahead of Link along the heading.
    let first = if direction == Direction::N { y - (y - 5).rem_euclid(8) } else { y + (5 - y).rem_euclid(8) };
    let mut t = None;
    for i in 0..3 {
        let g = first + i * step;
        if (g - y).abs() > 16 {
            break;
        }
        if at_dungeon_bound(x, g, direction) {
            return None;
        }
        if blocked(g) {
            t = Some(g);
            break;
        }
    }
    let t = t?;
    // A 16-pixel square: the grid point past T is stopped too; there the skip ends, inside the room bounds.
    let clipped = t + step;
    if !(UW_TOP..=UW_BOTTOM).contains(&clipped) || !blocked(clipped) {
        return None;
    }
    // Blocks, statues and walls only, never water: water ahead is the ladder clip's (or no clip without the ladder).
    let water = |g: i32| Room::hotspot_tiles(x, g, direction).into_iter().any(|(c, r)| room.tile(c, r) == WATER_TILE);
    if water(t) || water(clipped) {
        return None;
    }
    // G' (the back-off point) inside the room bounds; the run-up cells open both ways. Conservative: the macro can
    // always back off to G', so squares whose G' is past a bound or solid are not offered even where a Link on G
    // with the good fraction would clip (level 1 room $22's south block from below, G' = $C5; the statues of level 2
    // room $7D, 16 pixels apart).
    let (g, g2) = (t - step, t - 2 * step);
    if !(UW_TOP..=UW_BOTTOM).contains(&g2)
        || !room.can_link_move_from(x, g2, direction)
        || !room.can_link_move_from(x, g, direction)
        || !room.can_link_move_from(x, t, back)
        || !room.can_link_move_from(x, g, back)
    {
        return None;
    }
    // No stairs spot on the walk (G', G, T): the warp check would take Link down mid-macro (level 1 room $22: from
    // inside the diamond toward the south block, G' is the stairs).
    if [g2, g, t].into_iter().any(|p| on_stairs(room, x, p)) {
        return None;
    }
    // Useful: from inside the square a sideways move goes on, into a tile walled off from T (mechanics.md "Block
    // clip"). At T -+ 8 Link's hotspot is in the square's tile row next to T and a sideways test reads the one tile
    // beside it; when the same test from T is open too, Link stopped on T can already walk that way, and the clip
    // only leads back out (half a column off a square, an isolated statue).
    let sideways = [Direction::W, Direction::E].into_iter().any(|d| {
        !at_dungeon_bound(x, clipped, d) && room.can_link_move_from(x, clipped, d) && !room.can_link_move_from(x, t, d)
    });
    sideways.then_some(t)
}

/// The dock warp's block clip (round 36, docs/routes/d3first.md "The wrong warps"): N from the land below a raft dock's
/// row into the water on that row W of the dock (rooms 0x55 and 0x3f, game.yaml `raft_exits` `dock`), where the
/// recorder's whirlwind carries Link over the dock's trigger. The plain grid-point skip heading N (mechanics.md "Block
/// clip"): T is the grid point 8 pixels below the dock's row (Y `$85`), the clip ends on the row (Y `$7D`), facing N.
/// Legal exactly there: Link's X on the grid W of the dock, within 16 pixels below T; T and the row stopped, with
/// overworld water ahead (at 0x3f's `$58` the test's other tile is the dock's); G and G' (`$8D`, `$95`) open both
/// ways (land). The overworld offers no other N clip ([`skip_target`]); the session offers this one only on a
/// dock-warp step (`crate::Session::set_warp`).
pub fn dock_clip_at(room: &Room, x: i32, y: i32) -> Option<i32> {
    let (dock_x, row) = room.dock()?;
    let t = row + 8;
    if room.location.level != 0 || x & 7 != 0 || !(0..dock_x).contains(&x) || !(t..=t + 16).contains(&y) {
        return None;
    }
    let (n, s) = (Direction::N, Direction::S);
    let water = |g: i32| {
        !room.can_link_move_from(x, g, n)
            && Room::hotspot_tiles(x, g, n).into_iter().any(|(c, r)| OW_WATER_TILES.contains(&room.tile(c, r)))
    };
    let (g, g2) = (t + 8, t + 16);
    let land = |p: i32| room.can_link_move_from(x, p, n) && room.can_link_move_from(x, p, s);
    (water(t) && water(row) && land(g) && land(g2) && room.can_link_move_from(x, t, s)).then_some(t)
}

/// The dock clip for Link in `state` ([`dock_clip_at`], with Link's state as [`block_target`] reads it).
pub fn dock_clip_target(state: &GameView) -> Option<SkipTarget> {
    if state.mode() != MODE_PLAY || state.link.status != 0 || state.var(Var::LinkShoveDir) != 0 {
        return None;
    }
    let offset = state.var(Var::LinkGridOffset);
    let facing = state.var(Var::LinkDirection) as u8;
    if offset != 0 && !matches!(facing, 4 | 8) {
        return None;
    }
    let t = dock_clip_at(state.room(), state.link.position.x, state.link.position.y)?;
    Some(SkipTarget { direction: Direction::N, kind: SkipKind::Block, t: t as u8, end: -8 })
}

/// The dock warp's water spot of `room`: the X nearest the dock W of it whose clip is legal, and the dock's row
/// (0x55: (`$78`, `$7D`); 0x3f: (`$58`, `$7D`), "right beside the dock").
pub fn dock_spot(room: &Room) -> Option<(i32, i32)> {
    let (dock_x, row) = room.dock()?;
    (0..dock_x).rev().step_by(8).map(|x| x & !7).find(|&x| dock_clip_at(room, x, row + 16).is_some()).map(|x| (x, row))
}

/// Link at (x, y), on the grid, stands where a MOVE W or E along the land below the dock ends on a dock-warp step
/// while the clip is offered (`Executor::run_timed`): on [`dock_spot`]'s column, where [`dock_clip_at`] is legal.
pub fn on_dock_column(room: &Room, x: i32, y: i32) -> bool {
    dock_spot(room).is_some_and(|(column, _)| column == x) && dock_clip_at(room, x, y).is_some()
}

/// Link stands in the water on the dock's row W of the dock (where the clip ends), in play.
pub fn on_dock_spot(state: &GameView) -> bool {
    let Some((dock_x, row)) = state.room().dock() else { return false };
    let (x, y) = (state.link.position.x, state.link.position.y);
    state.mode() == MODE_PLAY && state.level() == 0 && state.link.status == 0 && y == row && (0..dock_x).contains(&x)
}

/// Link at rest on (x, y) stands on a dungeon stairs square, whose warp check takes him down (asm:Warp_CheckUnderLink:
/// X a multiple of $10, Y low nibble $D, the tile at (X, Y + $0B) one of $70-$73).
fn on_stairs(room: &Room, x: i32, y: i32) -> bool {
    x & 0xF == 0 && y & 0xF == 0xD && (0x70..=0x73).contains(&room.tile(x.div_euclid(8), (y - 53).div_euclid(8)))
}

/// What a tile test reads ahead of Link (the tiles `Room::hotspot_tiles` names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ahead {
    /// Every tile is water (`$F4`): a ladder is laid there, or a laid one is needed.
    Water,
    /// Every tile is walkable (doorway corridors included, ladder crossings not).
    Land,
    /// Anything else (a wall, a block, water beside land).
    Wall,
}

fn ahead(room: &Room, x: i32, y: i32, direction: Direction) -> Ahead {
    let tiles = Room::hotspot_tiles(x, y, direction);
    let inside = |&(c, r): &(i32, i32)| (0..crate::room::COLS).contains(&c) && (0..crate::room::ROWS).contains(&r);
    if tiles.iter().all(|t| inside(t) && room.is_water(room.tile(t.0, t.1))) {
        Ahead::Water
    } else if tiles.iter().all(|&(c, r)| room.is_tile_walkable(c, r)) {
        Ahead::Land
    } else {
        Ahead::Wall
    }
}

/// -1 heading W or N (toward smaller coordinates), +1 heading E or S.
fn sign(direction: Direction) -> i32 {
    if matches!(direction, Direction::W | Direction::N) { -1 } else { 1 }
}

/// The ladder clip from the lay point `lay` (Link's coordinate on the heading's axis where the first ladder is laid,
/// `across` his coordinate on the other axis), heading `direction`: (T, the macro's end in the heading's sense, the
/// ladders put away on the way), or None where the game does not carry Link over the water onto land with one press
/// (mechanics.md "Ladder clip", "Reach of one press").
///
/// - Up: the ladder is laid at the lay point L (G of the skip; G' = L + 8 is land), T = L - 8. Four tile rows of water
///   (two squares: the plain ladder stops at T, where case E finds water 8 pixels higher too), the relay lays a
///   second ladder at L - `$18`, and its distance-0 point L - `$20` must look onto land. Link ends at L - `$30`, the
///   second ladder put away (room `$21`: Y `$55`, 8 pixels past the top bound).
/// - Left, right, down: L is G' (the ladder lies one square ahead), T = L -+ `$10`, the square beyond it water too.
///   At the relay L -+ `$20` the square ahead is land (two squares: Link walks on to L -+ `$30`, inside the room
///   bounds) or water (three squares: a second ladder, whose distance-0 point L -+ `$30` must look onto land; Link
///   ends at L -+ `$40`).
///
/// Overworld (round 24, W and E only; mechanics.md "Overworld: room `$5F`"): the screen bounds instead of the
/// dungeon's, no border band (its mask leaves the perpendicular press alone, asm:Link_MaskInputInBorder), and L may
/// be on water: 0x5f's arrival from 0x60's wrap exit (X `$F0`), where Left lays the first ladder at `$E0`.
fn ladder_reach(room: &Room, across: i32, lay: i32, direction: Direction) -> Option<(i32, i8, u8)> {
    let horizontal = matches!(direction, Direction::W | Direction::E);
    let overworld = room.location.level == 0;
    if overworld && !horizontal {
        return None;
    }
    let s = sign(direction);
    let at = |a: i32, d: Direction| if horizontal { ahead(room, a, across, d) } else { ahead(room, across, a, d) };
    let water = |a: i32| at(a, direction) == Ahead::Water;
    let land = |a: i32| at(a, direction) == Ahead::Land;
    // Link stands on a grid point of the room's floor at the lay point.
    let ((left, right), (top, bottom)) =
        if overworld { ((OW_LEFT, OW_RIGHT), (OW_TOP, OW_BOTTOM)) } else { ((UW_LEFT, UW_RIGHT), (UW_TOP, UW_BOTTOM)) };
    let (lo, hi) = if horizontal { (left, right) } else { (top, bottom) };
    let (cross_lo, cross_hi) = if horizontal { (top, bottom) } else { (left, right) };
    let grid = |a: i32, horizontal_axis: bool| if horizontal_axis { a & 7 == 0 } else { a & 7 == 5 };
    if !(lo..=hi).contains(&lay)
        || !(cross_lo..=cross_hi).contains(&across)
        || !grid(lay, horizontal)
        || !grid(across, !horizontal)
    {
        return None;
    }
    // On land there: both tiles of his hotspot row walkable (he can't stand on water without a ladder), or on the
    // overworld's water where a room change put him.
    let (x, y) = if horizontal { (lay, across) } else { (across, lay) };
    let row = (y - 53).div_euclid(8);
    let floor = |c: i32| room.is_tile_walkable(c, row) || (overworld && room.is_water(room.tile(c, row)));
    if !floor(x.div_euclid(8)) || !floor(x.div_euclid(8) + 1) {
        return None;
    }
    let (t, end, stashes) = if direction == Direction::N {
        let t = lay - 8;
        let g2 = lay + 8;
        if g2 > UW_BOTTOM || [lay, t, lay - 16, lay - 24].into_iter().any(|a| !water(a)) || !land(lay - 32) {
            return None;
        }
        // The run-up cell G' to L and the back-off from L to G'.
        if !land(g2) || at(lay, Direction::S) != Ahead::Land {
            return None;
        }
        (t, lay - 48, 2)
    } else {
        let t = lay + 16 * s;
        if !water(lay) || !water(t) {
            return None;
        }
        let relay = lay + 32 * s;
        if land(relay) {
            // Two squares: walking on after the relay, inside the room bounds.
            let end = lay + 48 * s;
            if !land(lay + 40 * s) || !(lo..=hi).contains(&end) {
                return None;
            }
            (t, end, 1)
        } else if water(relay) && land(lay + 48 * s) && land(lay + 56 * s) {
            (t, lay + 64 * s, 2)
        } else {
            return None;
        }
    };
    // The press is not dropped: the trick position (5 short of T) outside the dungeon's outer border band on the
    // heading's axis (asm:Link_MaskInputInBorder).
    let trick = t - 5 * s;
    let band = overworld
        || match direction {
            Direction::W => trick >= UW_BAND_W,
            Direction::E => trick < UW_BAND_E,
            Direction::N => trick >= UW_BAND_N,
            _ => trick < UW_BAND_S,
        };
    band.then_some((t, ((t - end) * s) as i8, stashes))
}

/// The ladder clip heading `direction` from Link's state: he holds the ladder in a room where the game lays it
/// (`Room::ladder`), stands on the heading's line within 16 pixels of T (at the lay point or between G' and it with
/// no ladder out; on a ladder laid along the heading), and a monster slot is free for the first ladder
/// (asm:Obj_FindEmptyMonsterSlot).
fn ladder_target(state: &GameView, direction: Direction) -> Option<SkipTarget> {
    if !state.room().ladder() || state.var(Var::DoorwayDir) != 0 {
        return None;
    }
    let horizontal = matches!(direction, Direction::W | Direction::E);
    let (x, y) = (state.link.position.x, state.link.position.y);
    let (along, across) = if horizontal { (x, y) } else { (y, x) };
    let offset = state.var(Var::LinkGridOffset);
    let facing = state.var(Var::LinkDirection) as u8;
    let along_axis = if horizontal { matches!(facing, 1 | 2) } else { matches!(facing, 4 | 8) };
    if offset != 0 && !along_axis {
        return None;
    }
    let s = sign(direction);
    let ram = state.ram();
    let slot = ram[LADDER_SLOT] as usize;
    let lays: Vec<i32> = if slot != 0 {
        // A ladder out along the heading, Link on its line: T is its distance-0 point (ladder X, or ladder Y - 3).
        if slot > LAST_MONSTER_SLOT || ram[OBJ_DIRECTION.0 + slot] != direction.value() {
            return None;
        }
        let (lx, ly) = (ram[OBJ_POS_X.0 + slot] as i32, ram[OBJ_POS_Y.0 + slot] as i32);
        let t = match horizontal {
            true if ly == y + 3 => lx,
            false if lx == x => ly - 3,
            _ => return None,
        };
        vec![if direction == Direction::N { t + 8 } else { t - 16 * s }]
    } else {
        if !(1..=LAST_MONSTER_SLOT).any(|i| ram[OBJ_ID.0 + i] == 0) {
            return None;
        }
        if direction == Direction::N {
            // On or below G = L, at most 8 pixels: L is the grid point at or above Link, or the one above that.
            let g = along - (along - 5).rem_euclid(8);
            vec![g, g - 8]
        } else {
            vec![along]
        }
    };
    lays.into_iter().find_map(|lay| {
        let (t, end, _) = ladder_reach(state.room(), across, lay, direction)?;
        let p = (t - along) * s;
        (0..=16).contains(&p).then_some(SkipTarget { direction, kind: SkipKind::Ladder, t: t as u8, end })
    })
}

/// Frames of a skip from rest `p` pixels before T (16: G', 8: G with the good fraction, 0: on T, which backs off two
/// cells first) to its end `end`, rounded up to path-search tiles (8 pixels at 1.5 pixels a frame). `stashes`:
/// ladders put away on the way, a still frame each; a ladder clip also lays its first ladder in a still frame.
fn skip_cost(p: i32, end: i8, stashes: u8) -> i32 {
    let approach = match p {
        16.. => 7,
        8..=15 => 2,
        _ => 11 + 7,
    } + i32::from(stashes > 0);
    let crossing = ((5 - i32::from(end)) * 2 + 2) / 3 + i32::from(stashes);
    let frames = approach + 1 + crossing;
    (frames * 3 + 15) / 16
}

/// The skips of `room` as jumps of the path search (`Room::wavefront` with clips on): from every rest position
/// (grid point, offset 0) of a skip's zone to Link's tile at its end, at [`skip_cost`]. Dungeon rooms: the block
/// clip (N, S) everywhere, the ladder clip (any heading) while Link holds the ladder (`Room::ladder`). Overworld rooms:
/// the wrap exit (`Room::wrap_exit`), from G', G and the bound on its rows to the tile off the W edge where the W
/// exit's search starts; the wrap onto the side of the room a game.yaml `wrap_in` exit is on (`Room::wrap_in`, round
/// 24: 0x55's W third), to the landing tile at the opposite edge; and the ladder clip (W, E) in a room where the game
/// lays the ladder (round 24: 0x5f). Any other overworld wrap is not a jump: it ends one MOVE from the opposite edge,
/// so a jump would make the path to that edge's exit lead out of the entry edge, and the policy learned the wrap
/// without one (`docs/experiments/round21-gauntlet.md`).
pub fn skip_jumps(room: &Room) -> Vec<Jump> {
    let location = room.location;
    if location.in_cave || !room.is_loaded() {
        return Vec::new();
    }
    if location.level == 0 {
        let mut jumps = wrap_exit_jumps(room);
        jumps.extend(wrap_in_jumps(room));
        if room.ladder() {
            jumps.extend(ladder_jumps(room, [Direction::W, Direction::E], (OW_LEFT, OW_RIGHT), (OW_TOP, OW_BOTTOM)));
        }
        return jumps;
    }
    let tile = |x: i32, y: i32| Position::new(x, y).tile();
    let mut jumps = Vec::new();
    for x in (UW_LEFT..=UW_RIGHT).step_by(8) {
        for y in (UW_TOP..=UW_BOTTOM).step_by(8) {
            for direction in [Direction::N, Direction::S] {
                if let Some(t) = block_at(room, x, y, direction) {
                    let end = t + 8 * sign(direction);
                    let p = (y - t).abs();
                    jumps.push(Jump { from: tile(x, y), to: tile(x, end), direction, cost: skip_cost(p, -8, 0) });
                }
            }
        }
    }
    if room.ladder() {
        jumps.extend(ladder_jumps(room, CARDINALS, (UW_LEFT, UW_RIGHT), (UW_TOP, UW_BOTTOM)));
    }
    jumps
}

/// The ladder clip's jumps of `room` heading `directions`, lay points and lines within the bounds `(left, right)`
/// and `(top, bottom)` ([`skip_jumps`]).
fn ladder_jumps<const N: usize>(
    room: &Room,
    directions: [Direction; N],
    (left, right): (i32, i32),
    (top, bottom): (i32, i32),
) -> Vec<Jump> {
    let tile = |x: i32, y: i32| Position::new(x, y).tile();
    let mut jumps = Vec::new();
    for direction in directions {
        let horizontal = matches!(direction, Direction::W | Direction::E);
        let (lines, points) = if horizontal {
            ((top..=bottom).step_by(8), (left..=right).step_by(8))
        } else {
            ((left..=right).step_by(8), (top..=bottom).step_by(8))
        };
        for across in lines {
            for lay in points.clone() {
                let Some((t, end, stashes)) = ladder_reach(room, across, lay, direction) else {
                    continue;
                };
                let s = sign(direction);
                let at = |a: i32| if horizontal { tile(a, across) } else { tile(across, a) };
                let to = at(t - i32::from(end) * s);
                for p in [16, 8, 0] {
                    jumps.push(Jump { from: at(t - p * s), to, direction, cost: skip_cost(p, end, stashes) });
                }
            }
        }
    }
    jumps
}

/// The wrap exit's jumps of the overworld `room` ([`skip_jumps`]).
fn wrap_exit_jumps(room: &Room) -> Vec<Jump> {
    let Some((lo, hi)) = room.wrap_exit() else {
        return Vec::new();
    };
    let mut jumps = Vec::new();
    for row in lo..=hi {
        let y = GAMEPLAY_START_Y + 8 * row + 5;
        for p in [16, 8, 0] {
            let x = i32::from(OW_BOUND_E) - p;
            if wrap_at(room, x, y, Direction::E) == Some(SkipKind::WrapExit) {
                let from = Position::new(x, y).tile();
                jumps.push(Jump { from, to: (-1, row), direction: Direction::E, cost: skip_cost(p, -16, 0) });
            }
        }
    }
    jumps
}

/// The jumps of the overworld `room`'s `wrap_in` exit ([`skip_jumps`]): for a W exit the E wrap from G', G and the
/// bound on its rows to the landing at X `$00`, for an E exit the W wrap to X `$F0`, where that exit is one MOVE away.
fn wrap_in_jumps(room: &Room) -> Vec<Jump> {
    let Some((exit, (lo, hi))) = room.wrap_in() else {
        return Vec::new();
    };
    let wrap = exit.opposite();
    let (bound, landing) = if wrap == Direction::E { (OW_RIGHT, OW_LEFT) } else { (OW_LEFT, OW_RIGHT) };
    let mut jumps = Vec::new();
    for row in lo..=hi {
        let y = GAMEPLAY_START_Y + 8 * row + 5;
        for p in [16, 8, 0] {
            let x = bound - p * sign(wrap);
            if wrap_at(room, x, y, wrap) == Some(SkipKind::Wrap) {
                let (from, to) = (Position::new(x, y).tile(), Position::new(landing, y).tile());
                jumps.push(Jump { from, to, direction: wrap, cost: skip_cost(p, -16, 0) });
            }
        }
    }
    jumps
}

/// Why a skip ended without completing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClipAbort {
    /// Not available in the state the action started from: nothing was pressed.
    Unavailable,
    /// A hit, a room change or game over (the interrupt the option reports).
    Interrupted(Interrupt),
    /// The presses ran out (each missed press ends on T and backs off).
    Missed,
    /// Link stood still (something the decision table doesn't know held him).
    Stalled,
    /// [`MAX_SKIP_FRAMES`] passed.
    FrameCap,
}

impl ClipAbort {
    pub const fn name(self) -> &'static str {
        match self {
            ClipAbort::Unavailable => "unavailable",
            ClipAbort::Interrupted(_) => "interrupted",
            ClipAbort::Missed => "missed",
            ClipAbort::Stalled => "stalled",
            ClipAbort::FrameCap => "frame-cap",
        }
    }
}

/// What one skip did (in `crate::OptionResult` and `Change`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClipOutcome {
    pub direction: Direction,
    /// None when the skip was unavailable.
    pub kind: Option<SkipKind>,
    /// None: completed. A block or ladder clip whose crossing carries Link on into the next room (4:$21's `CLIP N`
    /// through its opened wall into $11) is completed: the room change ends the macro, but past the skipped point.
    pub abort: Option<ClipAbort>,
    /// Perpendicular presses made.
    pub presses: u8,
    /// Where the skip went (None when unavailable): what a MOVE must cross to finish an aborted one
    /// ([`OpenClip`]).
    pub target: Option<SkipTarget>,
}

impl ClipOutcome {
    pub fn completed(&self) -> bool {
        self.abort.is_none()
    }
}

/// A skip that started (the macro walked or pressed) and was cut short in its room: a MOVE along its heading that
/// crosses its square later counts as finishing it ([`ClipStats::finished_by_move`]; round 22). Kept by the session
/// until the skip is finished that way, Link leaves the room, or another CLIP starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenClip {
    pub target: SkipTarget,
    /// Link's coordinate across the heading (X for N/S, Y for W/E) where the skip was cut: the line it runs on.
    pub across: u8,
    pub location: crate::enums::MapLocation,
}

impl OpenClip {
    /// The skip of `outcome`, aborted in `state` (still in the room it started in), else None: completed, never
    /// started (unavailable), or cut by a room change or game over.
    pub fn of(outcome: &ClipOutcome, start: &GameView, state: &GameView) -> Option<OpenClip> {
        let target = outcome.target?;
        if outcome.completed() || start.full_location() != state.full_location() || state.game_over() {
            return None;
        }
        Some(OpenClip { target, across: across(state, target.direction), location: state.full_location() })
    }

    /// The MOVE from `prev` to `state` finishes this skip: it heads the skip's way, stays in its room and on its line,
    /// and ends at least as far past the skipped point as the macro's end point (in a block, beyond the square). A
    /// wrap exit is finished by the MOVE W from its line that leaves the room by its W edge (only a Link past the
    /// bound, X `$00`-`$07`, can: the edge is solid).
    pub fn finished_by(&self, action: &crate::actions::Action, prev: &GameView, state: &GameView) -> bool {
        let on_line = across(prev, self.target.direction) == self.across;
        if self.target.kind == SkipKind::WrapExit {
            return action.kind == ActionKind::Move
                && action.direction == Direction::W
                && on_line
                && prev.full_location() == self.location
                && self.location.direction_to(state.full_location()) == Direction::W;
        }
        action.kind == ActionKind::Move
            && action.direction == self.target.direction
            && state.full_location() == self.location
            && on_line
            && across(state, self.target.direction) == self.across
            && self.target.read(state.ram()).p <= self.target.end
    }
}

/// Link's coordinate across `heading`: X for N/S, Y for W/E.
fn across(state: &GameView, heading: Direction) -> u8 {
    let ram = state.ram();
    if matches!(heading, Direction::N | Direction::S) { ram[LINK_X] } else { ram[LINK_Y] }
}

/// Skips per episode, room visit or chain room ([`ClipStats::update`] per step).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClipStats {
    /// Skips completed (the macro reached its end point, or carried Link on into the next room).
    pub done: u32,
    /// Skips started that did not complete (a hit, a missed press, unavailable).
    pub aborted: u32,
    /// Completed by direction.
    pub done_n: u32,
    pub done_s: u32,
    pub done_w: u32,
    pub done_e: u32,
    /// Aborted skips a later MOVE finished by crossing their square (round 22; [`OpenClip`]): each is also in
    /// `aborted`, never in `done` (`done` counts the macro's own successes; the skip got done `done +
    /// finished_by_move` times).
    pub finished_by_move: u32,
}

impl ClipStats {
    pub fn update(&mut self, change: &Change) {
        if change.clip_finished_by_move {
            self.finished_by_move += 1;
        }
        let Some(clip) = change.clip else {
            return;
        };
        if !clip.completed() {
            self.aborted += 1;
            return;
        }
        self.done += 1;
        match clip.direction {
            Direction::N => self.done_n += 1,
            Direction::S => self.done_s += 1,
            Direction::W => self.done_w += 1,
            _ => self.done_e += 1,
        }
    }

    /// `{done, aborted, done_n, done_s, done_w, done_e, finished_by_move}`.
    pub fn fields(&self) -> [(&'static str, u32); 7] {
        [
            ("done", self.done),
            ("aborted", self.aborted),
            ("done_n", self.done_n),
            ("done_s", self.done_s),
            ("done_w", self.done_w),
            ("done_e", self.done_e),
            ("finished_by_move", self.finished_by_move),
        ]
    }
}

impl AddAssign for ClipStats {
    fn add_assign(&mut self, o: ClipStats) {
        self.done += o.done;
        self.aborted += o.aborted;
        self.done_n += o.done_n;
        self.done_s += o.done_s;
        self.done_w += o.done_w;
        self.done_e += o.done_e;
        self.finished_by_move += o.finished_by_move;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(p: i8, o: i8, good: bool) -> SkipState {
        SkipState { p, o, good, facing: true, ladder: false, raw: (0, 0, 0, 0) }
    }

    fn wrap_w() -> Controller {
        Controller::new(SkipTarget { direction: Direction::W, kind: SkipKind::Wrap, t: 0, end: -16 })
    }

    /// The left-edge frame table of mechanics.md: from rest on G = $08 with $80, Left 2, Up, then Left to $F0.
    #[test]
    fn decision_table_follows_the_traced_left_wrap() {
        let mut c = wrap_w();
        let target = c.target;
        let ram = |x: u8, o: u8, f: u8| {
            let mut r = [0u8; 0x800];
            r[0x70] = x;
            r[0x394] = o;
            r[LINK_POS_FRAC] = f;
            r[LINK_DIRECTION] = 2;
            r
        };
        let rows: [((u8, u8, u8), Decision); 6] = [
            ((0x08, 0x00, 0x80), Decision::Input(Input::Toward)),
            ((0x07, 0xFF, 0x00), Decision::Input(Input::Toward)),
            ((0x05, 0xFD, 0x80), Decision::Input(Input::Press)),
            ((0x07, 0x07, 0x00), Decision::Input(Input::Toward)),
            ((0xF8, 0x00, 0x00), Decision::Input(Input::Toward)),
            ((0xF0, 0x00, 0x80), Decision::Done),
        ];
        for (i, ((x, o, f), want)) in rows.into_iter().enumerate() {
            let s = target.read(&ram(x, o, f));
            assert_eq!(c.decide(s), want, "row {i}: {s:?}");
            if want == Decision::Input(Input::Press) {
                c.pressed(target.read(&ram(0x07, 0x07, 0x00)));
                assert_eq!(c.phase, Phase::Crossing);
            }
        }
    }

    #[test]
    fn wrong_cadence_and_stopped_on_t_back_off_to_g_prime() {
        // On G with the other fraction: walk away until G' ($10), then toward.
        let mut c = wrap_w();
        assert_eq!(c.decide(state(8, 0, false)), Decision::Input(Input::Away));
        assert_eq!(c.decide(state(12, -4, false)), Decision::Input(Input::Away));
        assert_eq!(c.decide(state(16, 0, true)), Decision::Input(Input::Toward));
        assert_eq!(c.phase, Phase::Approach);
        // On T (the bound, after entering from the left neighbour): never toward.
        let mut c = wrap_w();
        assert_eq!(c.decide(state(0, 0, false)), Decision::Input(Input::Away));
        // Between G and T in the wrong state: lost.
        for (p, o, good) in [(6, 2, false), (5, 3, false), (4, 4, true), (7, 1, true), (3, -3, true)] {
            let mut c = wrap_w();
            assert_eq!(c.decide(state(p, o, good)), Decision::Input(Input::Away), "{p} {o} {good}");
        }
        // Between G' and G either way, and farther back: toward.
        for (p, o) in [(12, 4), (12, -4), (16, 0), (32, 0), (20, 3)] {
            let mut c = wrap_w();
            assert_eq!(c.decide(state(p, o, false)), Decision::Input(Input::Toward), "{p} {o}");
        }
    }

    #[test]
    fn a_missed_crossing_backs_off_and_the_presses_run_out() {
        let mut c = wrap_w();
        assert_eq!(c.decide(state(5, 3, true)), Decision::Input(Input::Press));
        c.pressed(state(7, -7, false));
        assert_eq!(c.decide(state(1, -1, true)), Decision::Input(Input::Toward));
        // Ended on T: back off (never toward at the bound).
        assert_eq!(c.decide(state(0, 0, true)), Decision::Input(Input::Away));
        assert_eq!(c.phase, Phase::BackOff);
        c.presses = MAX_SKIP_PRESSES;
        c.phase = Phase::Approach;
        assert_eq!(c.decide(state(5, 3, true)), Decision::GiveUp);
    }

    #[test]
    fn a_press_lost_to_a_lag_frame_is_repeated() {
        let mut c = wrap_w();
        assert_eq!(c.decide(state(5, 3, true)), Decision::Input(Input::Press));
        c.pressed(state(5, 3, true));
        assert_eq!(c.phase, Phase::Approach);
        assert_eq!(c.decide(state(5, 3, true)), Decision::Input(Input::Press));
        assert_eq!(c.presses, 2);
    }

    #[test]
    fn right_and_down_headings_read_the_mirrored_state() {
        let t = SkipTarget { direction: Direction::E, kind: SkipKind::Wrap, t: 0xF0, end: -16 };
        let mut ram = [0u8; 0x800];
        (ram[0x70], ram[0x394], ram[LINK_POS_FRAC], ram[LINK_DIRECTION]) = (0xEB, 0x03, 0x00, 1);
        assert!(t.read(&ram).is_trick());
        ram[LINK_DIRECTION] = 2;
        assert!(!t.read(&ram).is_trick(), "walking away from T");
        (ram[0x70], ram[0x394]) = (0x00, 0x00);
        let s = t.read(&ram);
        assert_eq!((s.p, s.o), (-16, 0));
        let t = SkipTarget { direction: Direction::N, kind: SkipKind::Block, t: 0x95, end: -8 };
        (ram[0x84], ram[0x394], ram[LINK_POS_FRAC], ram[LINK_DIRECTION]) = (0x9A, 0xFD, 0x80, 8);
        assert!(t.read(&ram).is_trick());
        (ram[0x84], ram[0x394]) = (0x8D, 0x00);
        assert_eq!(t.read(&ram).p, t.end);
        let t = SkipTarget { direction: Direction::S, kind: SkipKind::Block, t: 0x7D, end: -8 };
        (ram[0x84], ram[0x394], ram[LINK_POS_FRAC], ram[LINK_DIRECTION]) = (0x78, 0x03, 0x00, 4);
        assert!(t.read(&ram).is_trick());
    }

    /// A ladder clip is done at its end point only once the last ladder is put away (the frame after arriving).
    #[test]
    fn a_ladder_clip_ends_with_the_ladder_put_away() {
        let mut c = Controller::new(SkipTarget { direction: Direction::N, kind: SkipKind::Ladder, t: 0x7D, end: -40 });
        c.phase = Phase::Crossing;
        let at_end = SkipState { p: -40, o: 0, good: true, facing: true, ladder: true, raw: (0x55, 0, 0x80, 8) };
        assert_eq!(c.decide(at_end), Decision::Input(Input::Release));
        assert_eq!(c.decide(SkipState { ladder: false, ..at_end }), Decision::Done);
        // Over the water at the relay (a grid point, the ladder out): on.
        assert_eq!(c.decide(SkipState { p: -16, ..at_end }), Decision::Input(Input::Toward));
    }

    /// A level 1 room in play: floor inside the walls, 16-pixel squares with their top-left pixel at `squares`
    /// (dungeon tile rows start at Y $40) and a stairs square (tiles $70-$73) at `stairs`, Link at rest on (x, y).
    fn dungeon_room(squares: &[(usize, usize)], stairs: Option<(usize, usize)>, x: u8, y: u8) -> GameView {
        dungeon_room_with(squares, &[], stairs, x, y, &[])
    }

    /// [`dungeon_room`] with water squares (tile $F4) at `water` and further RAM `vars`.
    fn dungeon_room_with(
        squares: &[(usize, usize)],
        water: &[(usize, usize)],
        stairs: Option<(usize, usize)>,
        x: u8,
        y: u8,
        vars: &[(Var, u8)],
    ) -> GameView {
        use crate::tables::{GRID_COLS, GRID_ROWS, RAM_SIZE, TILE_LAYOUT};
        let mut ram = [0u8; RAM_SIZE];
        let tile = |c: usize, r: usize| TILE_LAYOUT.0 + c * GRID_ROWS + r;
        for c in 0..GRID_COLS {
            for r in 0..GRID_ROWS {
                let wall = !(4..28).contains(&c) || !(4..18).contains(&r);
                ram[tile(c, r)] = if wall { 0xF6 } else { 0x74 };
            }
        }
        let square = |ram: &mut [u8; RAM_SIZE], (sx, sy): (usize, usize), first: u8, step: u8| {
            let (c, r) = (sx / 8, (sy - 0x40) / 8);
            for (i, (dc, dr)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                ram[tile(c + dc, r + dr)] = first + step * i as u8;
            }
        };
        for &at in squares {
            square(&mut ram, at, 0xB0, 0);
        }
        for &at in water {
            square(&mut ram, at, WATER_TILE, 0);
        }
        if let Some(at) = stairs {
            square(&mut ram, at, 0x70, 1);
        }
        let base = [(Var::Level, 1), (Var::Location, 0x22), (Var::Mode, MODE_PLAY), (Var::LinkX, x), (Var::LinkY, y)];
        for &(var, value) in base.iter().chain(vars) {
            ram[var.addr() as usize] = value;
        }
        GameView::new(&ram, 0, false, std::sync::Arc::new(crate::GameMap::builtin()))
    }

    /// The block clip goes into blocks, statues and walls, never into water: the pocket's square as water is not
    /// offered, without the ladder (water a wall) or with it (one square of water, which the ladder crosses: not
    /// even a stop). With a ladder out (`LadderSlot`) or in a doorway (`DoorwayDir`) nothing is offered.
    #[test]
    fn no_block_clip_into_water_with_a_ladder_out_or_from_a_doorway() {
        let pocket = [(0x70, 0x90), (0x90, 0x90)];
        let n = |view: &GameView| skip_target(view, Direction::N).map(|t| t.t);
        let block = dungeon_room_with(&[(0x80, 0x80), (0x70, 0x90), (0x90, 0x90)], &[], None, 0x80, 0x95, &[]);
        assert_eq!(n(&block), Some(0x85), "the pocket's block");
        for ladder in [0, 1] {
            let view = dungeon_room_with(&pocket, &[(0x80, 0x80)], None, 0x80, 0x95, &[(Var::StepLadder, ladder)]);
            assert_eq!(n(&view), None, "water, ladder {ladder}");
        }
        for (var, value) in [(Var::LadderSlot, 0x0B), (Var::DoorwayDir, 8)] {
            let view = dungeon_room_with(&[(0x80, 0x80), (0x70, 0x90), (0x90, 0x90)], &[], None, 0x80, 0x95, &[(var, value)]);
            assert!(CARDINALS.into_iter().all(|d| !skip_available(&view, d)), "{var:?} {value}");
        }
    }

    /// "Useful clips only": a block clip is offered where, from 8 pixels into the square, a sideways move enters a
    /// tile that Link stopped on T can't step into (mechanics.md "Block clip"). Not with no open sideways tile, and
    /// not past an isolated square, where the sideways tile is as open from T and the clip only leads back out. Not
    /// where the walk G'..T passes a stairs spot either (the warp would take Link down mid-macro).
    #[test]
    fn a_block_clip_is_offered_only_where_it_leads_somewhere() {
        let target = |squares: &[(usize, usize)], y: u8, direction: Direction| {
            skip_target(&dungeon_room(squares, None, 0x80, y), direction).map(|t| t.t)
        };
        // Heading N under the square ($80, $80): T = $85, G' = $95. The squares below-left and below-right wall off
        // the tiles beside T; from T - 8 = $7D both sides are open.
        let pocket = [(0x80, 0x80), (0x70, 0x90), (0x90, 0x90)];
        for y in [0x95, 0x8D, 0x85] {
            assert_eq!(target(&pocket, y, Direction::N), Some(0x85), "pocket, Y {y:#04x}");
            assert_eq!(target(&pocket, y, Direction::W), None, "no W/E block clip");
            assert_eq!(target(&pocket, y, Direction::E), None, "no W/E block clip");
        }
        assert_eq!(target(&pocket, 0x9D, Direction::N), None, "24 pixels from T: outside the zone");
        // Stairs under G = $8D (the square at ($80, $90), between the lower squares): no clip from the zone.
        let view = dungeon_room(&pocket, Some((0x80, 0x90)), 0x80, 0x95);
        assert!(view.room().stairs_tiles() == vec![(0x10, 0x0A)] && skip_target(&view, Direction::N).is_none());
        // No open sideways tile at the clipped position: squares beside the square on both sides.
        let walled = [(0x70, 0x80), (0x80, 0x80), (0x90, 0x80), (0x70, 0x90), (0x90, 0x90)];
        assert_eq!(target(&walled, 0x95, Direction::N), None, "walled in");
        // One open side is enough.
        let one_side = [(0x70, 0x80), (0x80, 0x80), (0x70, 0x90), (0x90, 0x90)];
        assert_eq!(target(&one_side, 0x95, Direction::N), Some(0x85), "E open");
        // An isolated square (a statue): the tiles beside T are open as well; the clip leads back out.
        assert_eq!(target(&[(0x80, 0x80)], 0x95, Direction::N), None, "isolated square");
        // Heading S over the square ($80, $90), the pocket mirrored: T = $7D, G' = $6D.
        let pocket_s = [(0x80, 0x90), (0x70, 0x80), (0x90, 0x80)];
        for y in [0x6D, 0x75, 0x7D] {
            assert_eq!(target(&pocket_s, y, Direction::S), Some(0x7D), "pocket S, Y {y:#04x}");
        }
        assert_eq!(target(&[(0x80, 0x90)], 0x6D, Direction::S), None, "isolated square from above");
    }

    /// The ladder clip's reach (mechanics.md "Reach of one press"): up over exactly two squares of water with land
    /// beyond (one square is the plain ladder's, three need a second press); left, right or down over two squares
    /// (Link walks on after the relay) or three (a second ladder), with land beyond. Offered only with the ladder,
    /// within 16 pixels of T, and its jumps start at every rest position of the zone.
    #[test]
    fn the_ladder_clip_crosses_two_or_three_squares_onto_land() {
        let ladder = [(Var::StepLadder, 1)];
        let room = |water: &[(usize, usize)]| dungeon_room_with(&[], water, None, 0x78, 0x8D, &ladder);
        let reach = |water: &[(usize, usize)], across: i32, lay: i32, d: Direction| {
            ladder_reach(room(water).room(), across, lay, d)
        };
        // Up from L = $85 (tile row 9 ahead): rows 6-9 water, row 5 land. Link ends at $55.
        let two = [(0x78, 0x80), (0x78, 0x70)];
        assert_eq!(reach(&two, 0x78, 0x85, Direction::N), Some((0x7D, -40, 2)));
        assert_eq!(reach(&two[..1], 0x78, 0x85, Direction::N), None, "one square: the plain ladder");
        assert_eq!(reach(&[(0x78, 0x80), (0x78, 0x70), (0x78, 0x60)], 0x78, 0x85, Direction::N), None, "three");
        for y in [0x8D, 0x85, 0x89] {
            let view = dungeon_room_with(&[], &two, None, 0x78, y, &ladder);
            let t = skip_target(&view, Direction::N).expect("in the zone");
            assert_eq!((t.kind, t.t, t.end), (SkipKind::Ladder, 0x7D, -40), "Y {y:#04x}");
        }
        assert!(skip_target(&dungeon_room_with(&[], &two, None, 0x78, 0x95, &ladder), Direction::N).is_none());
        assert!(skip_target(&dungeon_room_with(&[], &two, None, 0x78, 0x8D, &[]), Direction::N).is_none(), "no ladder");
        // Left from L = $60 on row Y $8D (tile row 11): two squares, then land: Link ends at $30.
        let w2 = [(0x50, 0x90), (0x40, 0x90)];
        assert_eq!(reach(&w2, 0x8D, 0x60, Direction::W), Some((0x50, -32, 1)));
        let w3 = [(0x50, 0x90), (0x40, 0x90), (0x30, 0x90)];
        assert_eq!(reach(&w3, 0x8D, 0x60, Direction::W), Some((0x50, -48, 2)));
        let w4 = [(0x50, 0x90), (0x40, 0x90), (0x30, 0x90), (0x20, 0x90)];
        assert_eq!(reach(&w4, 0x8D, 0x60, Direction::W), None, "four squares");
        // The jumps: N from G', G and T to Link's tile at $55; W from L, G and T to his tile at $30 (on rows Y $85 and
        // $8D, whose feet row is the water's; X $58 is no lay point: Link would stand half on the water).
        let jumps = skip_jumps(room(&two).room());
        let n: Vec<_> = jumps.iter().filter(|j| j.direction == Direction::N).map(|j| (j.from, j.to)).collect();
        assert_eq!(n, [((15, 10), (15, 3)), ((15, 9), (15, 3)), ((15, 8), (15, 3))]);
        let jumps = skip_jumps(room(&w2).room());
        let w: Vec<_> = jumps.iter().filter(|j| j.direction == Direction::W).map(|j| (j.from, j.to)).collect();
        assert_eq!(w.len(), 6, "{w:?}");
        for want in [((12, 10), (6, 10)), ((11, 10), (6, 10)), ((10, 10), (6, 10)), ((12, 9), (6, 9))] {
            assert!(w.contains(&want), "{want:?} in {w:?}");
        }
        assert!(skip_jumps(dungeon_room_with(&[], &two, None, 0x78, 0x8D, &[]).room()).is_empty(), "no ladder");
    }

    /// A wrap exit turns back on the landing frame (X `$00`, offset 0) and holds Left from then on.
    #[test]
    fn a_wrap_exit_turns_back_on_the_landing_frame() {
        let mut c = Controller::new(SkipTarget { direction: Direction::E, kind: SkipKind::WrapExit, t: 0xF0, end: -16 });
        assert_eq!(c.decide(state(5, 3, true)), Decision::Input(Input::Press));
        c.pressed(state(7, -7, false));
        assert_eq!(c.decide(state(-8, 0, true)), Decision::Input(Input::Toward), "X $F8: on");
        assert_eq!(c.decide(state(-15, 7, false)), Decision::Input(Input::Toward), "X $FF: on");
        assert_eq!(c.decide(state(-16, 0, true)), Decision::Input(Input::Away), "landed: Left");
        assert_eq!(c.phase, Phase::Exit);
        assert_eq!(c.decide(state(-16, 0, true)), Decision::Input(Input::Away), "still Left");
    }

    /// An overworld room in play: every tile walkable but columns 0-1 (trees, every row), Link at rest on (x, y)
    /// facing E.
    fn overworld_room(location: u8, x: u8, y: u8) -> GameView {
        use crate::tables::{GRID_COLS, GRID_ROWS, RAM_SIZE, TILE_LAYOUT};
        let mut ram = [0u8; RAM_SIZE];
        for c in 0..GRID_COLS {
            for r in 0..GRID_ROWS {
                ram[TILE_LAYOUT.0 + c * GRID_ROWS + r] = if c < 2 { 0xC5 } else { 0x26 };
            }
        }
        let vars = [(Var::Level, 0), (Var::Location, location), (Var::Mode, MODE_PLAY), (Var::LinkX, x), (Var::LinkY, y)];
        for (var, value) in vars {
            ram[var.addr() as usize] = value;
        }
        ram[LINK_DIRECTION] = 1;
        GameView::new(&ram, 0, false, std::sync::Arc::new(crate::GameMap::builtin()))
    }

    /// CLIP E onto the solid left edge is the wrap exit only in a room with a game.yaml W `wrap` exit (0x75) and only on
    /// its rows (Link's tile rows 5-14, Y `$65`-`$AD`); never W; the path search jumps from G', G and the bound on
    /// those rows off the W edge, and with clips a W target is reached from the zone (without clips not at all).
    #[test]
    fn the_wrap_exit_is_offered_only_on_its_rows() {
        let kind = |room: u8, x: u8, y: u8, d: Direction| skip_target(&overworld_room(room, x, y), d).map(|t| t.kind);
        for y in (0x3D..=0xDD).step_by(8) {
            let safe = (0x65..=0xAD).contains(&y);
            let want = safe.then_some(SkipKind::WrapExit);
            assert_eq!(kind(0x75, 0xE0, y as u8, Direction::E), want, "0x75 Y {y:#04x}");
            assert_eq!(kind(0x75, 0xD0, y as u8, Direction::E), want, "0x75 Y {y:#04x} from $D0");
            assert_eq!(kind(0x77, 0xE0, y as u8, Direction::E), None, "0x77 has no wrap exit: Y {y:#04x}");
            assert_eq!(kind(0x75, 0x10, y as u8, Direction::W), None, "W wrap onto open landing, run-up solid");
        }
        assert_eq!(kind(0x75, 0xC8, 0x8D, Direction::E), None, "outside the zone");
        assert_eq!(kind(0x75, 0xE0, 0x8E, Direction::E), None, "not on a row line");
        let view = overworld_room(0x75, 0xE0, 0x8D);
        let room = view.room();
        assert_eq!(room.wrap_exit(), Some((5, 14)));
        let jumps = skip_jumps(room);
        assert_eq!(jumps.len(), 30, "3 rest positions x 10 rows: {jumps:?}");
        assert!(jumps.iter().all(|j| j.direction == Direction::E && j.to == (-1, j.from.1)), "{jumps:?}");
        let from: std::collections::BTreeSet<(i32, i32)> = jumps.iter().map(|j| j.from).collect();
        assert!(from.contains(&(28, 5)) && from.contains(&(30, 14)) && !from.contains(&(28, 4)), "{from:?}");
        let w = [crate::room::Target::Dir(Direction::W)];
        let with = room.wavefront(&w, &[], &[], true);
        assert!(with.get(view.link.tile()).is_some_and(|d| d > 0), "the zone leads out W");
        assert!(with.get((20, 3)).is_some(), "so does the rest of the E part");
        assert_eq!(room.wavefront(&w, &[], &[], false).get(view.link.tile()), None, "no clips: no way out W");
    }
}
