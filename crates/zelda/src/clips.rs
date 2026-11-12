//! The grid-point skip as an option (round 21; `zelda-asm/mechanics.md`, "# Screen wrap (overworld)" and
//! "# Grid-point skip: block clip, ladder clip and their run-up").
//!
//! One frame of perpendicular input 5 pixels short of a grid point T, in the right sub-pixel cadence, lets Link cross
//! T with no frame starting on it, so T's tile and screen-edge tests are skipped. The agent's MOVE (4-32 frames)
//! can't do that, so the executor plays it as a macro ([`Controller`], driven by `crate::executor`): it reads
//! `$70`/`$84`, `$394` and `$3A8` every frame and follows the decision table of mechanics.md (back off to G' when too
//! close or in the wrong cadence, walk to the trick state, press one perpendicular frame, hold the heading). No RAM
//! is written.
//!
//! Two uses: the overworld **wrap** (W, E): T is the screen bound, Link comes in at the opposite side of the same
//! room and the macro walks on to where that side's ordinary room change is one MOVE away (X `$F0` after a W wrap,
//! `$00` after an E wrap); and the dungeon **block clip** (N, S): T is where a 16-pixel solid square stops Link, the
//! skip carries him 8 pixels into it (T -+ 8), from where a sideways MOVE goes on.
//!
//! [`skip_target`] (state only) is the single availability rule of the mask, the executor and the observation.
//! [`ClipSettings`] (the specialist's `clips.action`, set on the session every step) says how the policy reaches the
//! macro: the CLIP kind, the longest MOVE, or a MOVE out of an edge that is not a correct exit.

use crate::actions::{Action, ActionMask, CARDINALS};
use crate::change::Change;
use crate::enums::{ActionKind, Direction};
use crate::executor::Interrupt;
use crate::game::{GameView, at_dungeon_bound};
use crate::room::Room;
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
/// Dungeon room bounds Link can stand on (asm:Room_ObjBoundsUW: Y $5D..$BD).
const UW_TOP: i32 = 0x5D;
const UW_BOTTOM: i32 = 0xBD;
/// Dungeon outer border band on X (asm:Link_BorderBounds): a perpendicular press is dropped there.
const UW_BAND_W: i32 = 0x17;
const UW_BAND_E: i32 = 0xD9;
/// Play mode.
const MODE_PLAY: u8 = 5;

/// `ObjPosFrac` of Link (`$3A8`): the sub-pixel that sets the 1-2 pixel cadence.
const LINK_POS_FRAC: usize = Var::LinkPosFrac.addr() as usize;
const LINK_X: usize = Var::LinkX.addr() as usize;
const LINK_Y: usize = Var::LinkY.addr() as usize;
const LINK_GRID_OFFSET: usize = Var::LinkGridOffset.addr() as usize;
const LINK_DIRECTION: usize = Var::LinkDirection.addr() as usize;

/// Frames the macro may take before it gives up (a run-up from 32 pixels with two back-offs takes about 90).
pub const MAX_SKIP_FRAMES: u32 = 200;
/// Presses the macro may make before it gives up (each failed one costs a back-off).
pub const MAX_SKIP_PRESSES: u8 = 3;
/// Frames in a row Link may stand still (no lag frame repeats this often) before the macro gives up.
pub const MAX_STILL_FRAMES: u32 = 4;

/// What the skip does in the direction it is offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkipKind {
    /// Overworld screen wrap (W, E).
    Wrap,
    /// Dungeon block clip (N, S).
    Block,
}

impl SkipKind {
    pub const fn name(self) -> &'static str {
        match self {
            SkipKind::Wrap => "wrap",
            SkipKind::Block => "block",
        }
    }
}

/// Where the skip goes: the heading, T on the heading's axis and where the macro ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkipTarget {
    pub direction: Direction,
    pub kind: SkipKind,
    /// The grid point whose tests are skipped (X for W/E, Y for N/S).
    pub t: u8,
}

impl SkipTarget {
    /// Distance still to go to T in the heading's sense when the macro is done (negative: past T). A wrap ends 16
    /// pixels past the bound (X `$F0` after a W wrap, `$00` after an E wrap); a block clip 8 pixels into the square.
    pub const fn end(&self) -> i8 {
        match self.kind {
            SkipKind::Wrap => -16,
            SkipKind::Block => -8,
        }
    }

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
        SkipState { p, o: o_h, good, facing, raw: (c, ram[LINK_GRID_OFFSET], f, ram[LINK_DIRECTION]) }
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
}

/// One frame's input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Toward,
    Away,
    Press,
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
        let end = self.target.end();
        match self.phase {
            Phase::Crossing => {
                if s.p == end && s.o == 0 {
                    return Decision::Done;
                }
                if s.p == 0 && s.o == 0 {
                    // Stopped on T: the press was missed (the edge or the square's test was made). Never hold the
                    // heading here (at a screen bound that is the ordinary room change): back off and try again.
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
/// "Availability"). Overworld W/E: the wrap; dungeon N/S: the block clip when it is useful. None elsewhere.
pub fn skip_target(state: &GameView, direction: Direction) -> Option<SkipTarget> {
    if state.mode() != MODE_PLAY
        || state.link.status != 0
        || state.var(Var::LinkShoveDir) != 0
        || state.var(Var::LadderSlot) != 0
    {
        return None;
    }
    match (state.level(), direction) {
        (0, Direction::W | Direction::E) => wrap_target(state, direction),
        (1.., Direction::N | Direction::S) => block_target(state, direction),
        _ => None,
    }
}

pub fn skip_available(state: &GameView, direction: Direction) -> bool {
    skip_target(state, direction).is_some()
}

fn wrap_target(state: &GameView, direction: Direction) -> Option<SkipTarget> {
    let (x, y) = (state.link.position.x, state.link.position.y);
    let offset = state.var(Var::LinkGridOffset);
    let facing = state.var(Var::LinkDirection) as u8;
    // On a row line: Y a grid point, and the offset (if any) along the row.
    if y & 7 != 5 || !(0x3D..=0xDD).contains(&y) || (offset != 0 && !matches!(facing, 1 | 2)) {
        return None;
    }
    let (t, zone, run_up, landing): (u8, bool, [i32; 4], [i32; 2]) = match direction {
        Direction::W => (OW_BOUND_W, x <= WRAP_ZONE as i32, [0, 1, 2, 3], [0x1E, 0x1F]),
        _ => (OW_BOUND_E, (0xD0..=OW_BOUND_E as i32).contains(&x), [0x1C, 0x1D, 0x1E, 0x1F], [0, 1]),
    };
    if !zone {
        return None;
    }
    // Every tile the macro walks over on the row (G' to the bound, the back-off included) and the landing cells at
    // the opposite edge (mechanics.md "After the wrap, edge by edge"; a solid one strands Link).
    let room = state.room();
    let row = (y - 53).div_euclid(8);
    let open = |c: i32| room.is_tile_walkable(c, row) && !SLOW_TILES.contains(&room.tile(c, row));
    if !run_up.into_iter().chain(landing).all(open) {
        return None;
    }
    Some(SkipTarget { direction, kind: SkipKind::Wrap, t })
}

fn block_target(state: &GameView, direction: Direction) -> Option<SkipTarget> {
    let (x, y) = (state.link.position.x, state.link.position.y);
    let offset = state.var(Var::LinkGridOffset);
    let facing = state.var(Var::LinkDirection) as u8;
    if state.var(Var::DoorwayDir) != 0
        || x & 7 != 0
        || !(UW_BAND_W..UW_BAND_E).contains(&x)
        || (offset != 0 && !matches!(facing, 4 | 8))
    {
        return None;
    }
    let room = state.room();
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
    if !sideways {
        return None;
    }
    Some(SkipTarget { direction, kind: SkipKind::Block, t: t as u8 })
}

/// Link at rest on (x, y) stands on a dungeon stairs square, whose warp check takes him down (asm:Warp_CheckUnderLink:
/// X a multiple of $10, Y low nibble $D, the tile at (X, Y + $0B) one of $70-$73).
fn on_stairs(room: &Room, x: i32, y: i32) -> bool {
    x & 0xF == 0 && y & 0xF == 0xD && (0x70..=0x73).contains(&room.tile(x.div_euclid(8), (y - 53).div_euclid(8)))
}

/// How the policy reaches the skip (the specialist's `clips.action`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ClipTrigger {
    /// No skip anywhere: CLIP is masked and MOVE is plain.
    #[default]
    None,
    /// The CLIP kind, legal where the skip is available.
    Kind,
    /// A MOVE of the longest duration (32) where the skip is available.
    Long,
    /// A MOVE of any duration where the skip is available and its direction is not a correct exit of the current
    /// objective (a block clip has no exit side: any MOVE into the square).
    Edge,
}

impl ClipTrigger {
    pub const ALL: [ClipTrigger; 4] = [ClipTrigger::None, ClipTrigger::Kind, ClipTrigger::Long, ClipTrigger::Edge];

    pub const fn name(self) -> &'static str {
        match self {
            ClipTrigger::None => "none",
            ClipTrigger::Kind => "kind",
            ClipTrigger::Long => "long",
            ClipTrigger::Edge => "edge",
        }
    }

    pub fn from_name(name: &str) -> Option<ClipTrigger> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }
}

/// The session's clip settings, set every step by the env (the trigger from the config, the correct exits from the
/// objective).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ClipSettings {
    pub trigger: ClipTrigger,
    /// Direction bits (E 1, W 2, S 4, N 8) of the current objective's correct exits: the `edge` trigger leaves them
    /// plain.
    pub correct_exits: u8,
}

impl ClipSettings {
    pub fn new(trigger: ClipTrigger, correct_exits: u8) -> ClipSettings {
        ClipSettings { trigger, correct_exits }
    }

    /// Whether a MOVE in `direction` (of any duration) triggers the skip under `edge`, given its target.
    fn edge_skips(&self, target: &SkipTarget) -> bool {
        target.kind == SkipKind::Block || self.correct_exits & target.direction.value() == 0
    }

    /// The skip a MOVE `action` performs in `state` instead of the plain walk (`long`, `edge`).
    pub fn move_target(&self, state: &GameView, action: &Action) -> Option<SkipTarget> {
        if action.kind != ActionKind::Move {
            return None;
        }
        match self.trigger {
            ClipTrigger::Long if action.duration == crate::actions::DURATIONS[3] => skip_target(state, action.direction),
            ClipTrigger::Edge => skip_target(state, action.direction).filter(|t| self.edge_skips(t)),
            _ => None,
        }
    }

    /// The skip (if any) `action` performs: a CLIP's always (whatever the trigger; the mask gates the policy), a
    /// MOVE's per [`ClipSettings::move_target`].
    pub fn action_target(&self, state: &GameView, action: &Action) -> Option<Option<SkipTarget>> {
        match action.kind {
            ActionKind::Clip => Some(skip_target(state, action.direction)),
            ActionKind::Move => self.move_target(state, action).map(Some),
            _ => None,
        }
    }

    /// Adds the trigger's legal actions to `mask` (computed from the state alone, CLIP never in it): CLIP in every
    /// direction the skip is available (`kind`); MOVE in a direction whose MOVE performs the skip (`long`, `edge`),
    /// so a MOVE into a square from the block-clip zone is legal.
    pub fn apply(&self, mask: &mut ActionMask, state: &GameView) {
        for d in CARDINALS {
            match self.trigger {
                ClipTrigger::None => {}
                ClipTrigger::Kind => {
                    if skip_available(state, d) {
                        mask.allow(ActionKind::Clip, Some(d));
                    }
                }
                ClipTrigger::Long => {
                    if skip_available(state, d) {
                        mask.allow(ActionKind::Move, Some(d));
                    }
                }
                ClipTrigger::Edge => {
                    if skip_target(state, d).is_some_and(|t| self.edge_skips(&t)) {
                        mask.allow(ActionKind::Move, Some(d));
                    }
                }
            }
        }
    }
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
    /// None: completed.
    pub abort: Option<ClipAbort>,
    /// Perpendicular presses made.
    pub presses: u8,
}

impl ClipOutcome {
    pub fn completed(&self) -> bool {
        self.abort.is_none()
    }
}

/// Skips per episode, room visit or chain room ([`ClipStats::update`] per step).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClipStats {
    /// Skips completed.
    pub done: u32,
    /// Skips started that did not complete (a hit, a missed press, unavailable).
    pub aborted: u32,
    /// Completed by direction.
    pub done_n: u32,
    pub done_s: u32,
    pub done_w: u32,
    pub done_e: u32,
}

impl ClipStats {
    pub fn update(&mut self, change: &Change) {
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

    /// `{done, aborted, done_n, done_s, done_w, done_e}`.
    pub fn fields(&self) -> [(&'static str, u32); 6] {
        [
            ("done", self.done),
            ("aborted", self.aborted),
            ("done_n", self.done_n),
            ("done_s", self.done_s),
            ("done_w", self.done_w),
            ("done_e", self.done_e),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(p: i8, o: i8, good: bool) -> SkipState {
        SkipState { p, o, good, facing: true, raw: (0, 0, 0, 0) }
    }

    fn wrap_w() -> Controller {
        Controller::new(SkipTarget { direction: Direction::W, kind: SkipKind::Wrap, t: 0 })
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
        let t = SkipTarget { direction: Direction::E, kind: SkipKind::Wrap, t: 0xF0 };
        let mut ram = [0u8; 0x800];
        (ram[0x70], ram[0x394], ram[LINK_POS_FRAC], ram[LINK_DIRECTION]) = (0xEB, 0x03, 0x00, 1);
        assert!(t.read(&ram).is_trick());
        ram[LINK_DIRECTION] = 2;
        assert!(!t.read(&ram).is_trick(), "walking away from T");
        (ram[0x70], ram[0x394]) = (0x00, 0x00);
        let s = t.read(&ram);
        assert_eq!((s.p, s.o), (-16, 0));
        let t = SkipTarget { direction: Direction::N, kind: SkipKind::Block, t: 0x95 };
        (ram[0x84], ram[0x394], ram[LINK_POS_FRAC], ram[LINK_DIRECTION]) = (0x9A, 0xFD, 0x80, 8);
        assert!(t.read(&ram).is_trick());
        (ram[0x84], ram[0x394]) = (0x8D, 0x00);
        assert_eq!(t.read(&ram).p, t.end());
        let t = SkipTarget { direction: Direction::S, kind: SkipKind::Block, t: 0x7D };
        (ram[0x84], ram[0x394], ram[LINK_POS_FRAC], ram[LINK_DIRECTION]) = (0x78, 0x03, 0x00, 4);
        assert!(t.read(&ram).is_trick());
    }

    #[test]
    fn triggers_round_trip_by_name() {
        for t in ClipTrigger::ALL {
            assert_eq!(ClipTrigger::from_name(t.name()), Some(t));
        }
        assert_eq!(ClipTrigger::from_name("sword"), None);
    }

    /// A level 1 room in play: floor inside the walls, 16-pixel squares with their top-left pixel at `squares`
    /// (dungeon tile rows start at Y $40) and a stairs square (tiles $70-$73) at `stairs`, Link at rest on (x, y).
    fn dungeon_room(squares: &[(usize, usize)], stairs: Option<(usize, usize)>, x: u8, y: u8) -> GameView {
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
        if let Some(at) = stairs {
            square(&mut ram, at, 0x70, 1);
        }
        let vars = [(Var::Level, 1), (Var::Location, 0x22), (Var::Mode, MODE_PLAY), (Var::LinkX, x), (Var::LinkY, y)];
        for (var, value) in vars {
            ram[var.addr() as usize] = value;
        }
        GameView::new(&ram, 0, false, std::sync::Arc::new(crate::GameMap::builtin()))
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
}
