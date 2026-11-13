//! Attacks played with the controller alone: how a player strikes in a direction from where Link stands.
//!
//! The game turns Link only through its own input rules (zelda-asm `mechanics.md` "Link's movement: grid, speed and
//! sub-pixels"; asm:Link_HandleInput), so an attack in direction `d` is a short button sequence:
//!
//! - **Sword (A).** A is read before the direction rules of the same frame (asm:Link_HandleInput) and the sword
//!   takes Link's facing every frame (asm:Sword_UpdateSwordOrRod), so `d`+A on one frame turns Link and swings `d`
//!   without moving him: no turn frame. Off a grid point a perpendicular press does not turn him (the six cases of
//!   asm:Link_ModifyDirOnGridLine): holding `d` lets the game walk him to the grid point (case 5 ahead, case 6 back to
//!   the nearer one), where the `d`+A frame follows.
//! - **Bomb, arrow (B).** They take Link's facing when B is read (asm:Link_PlaceWeaponAhead, asm:Link_WieldWeapon),
//!   so Link must face `d` first: one frame of `d` turns him on a grid point (and moves him 1-2 pixels if the way
//!   is open), then B.
//! - **Boomerang (B).** It flies along the held directions (asm:Boomerang_Wield), diagonals too: `d`+B on one frame.
//! - **Border bands.** asm:Link_FilterInput drops A and B by the coordinate on the axis Link faces when the button
//!   is pressed ([`button_survives`]). When the facing's band drops the button Link turns first (one frame).
//! - **Doorways.** With `DoorwayDir` ($53) set the game forces the input onto Link's facing axis
//!   (asm:Link_ModifyDirInDoorway): a sideways press walks him forward. Facing into the room, holding the sideways
//!   direction walks him back onto the room's edge, where the game turns him; facing out of the room it would walk
//!   him through the door, so that attack is impossible.
//! - **Held Link.** A Like-Like's hold (asm:IsLinkParalyzed) discards the directions and knockback
//!   (asm:ObjShoveDir) skips the direction rules: only the facing is left (and the boomerang's held direction
//!   under knockback).
//!
//! [`next`] decides the next frame's input from Link's RAM; the executor plays it frame by frame on the real game
//! and [`plan`] plays it on a model of Link's input handling ([`step`]) to tell the action mask whether the attack
//! can be made. A sequence whose input would leave the room, enter a doorway or open a door is not an attack and is
//! refused ([`plan`] returns `None`).

use crate::enums::{ActionKind, Direction};
use crate::executor::{MODE_CAVE, buttons, direction_buttons};
use crate::game::{GameView, MODE_CELLAR};

const ADDR_LEVEL: usize = 0x10;
const ADDR_MODE: usize = 0x12;
const ADDR_DOORWAY_DIR: usize = 0x53;
const ADDR_GO_STRAIGHT: usize = 0x57;
const ADDR_LADDER_SLOT: usize = 0x64;
const ADDR_LINK_X: usize = 0x70;
const ADDR_LINK_Y: usize = 0x84;
const ADDR_LINK_DIR: usize = 0x98;
const ADDR_LINK_STATE: usize = 0xAC;
const ADDR_SHOVE_DIR: usize = 0xC0;
const ADDR_GRID_OFFSET: usize = 0x394;
const ADDR_POS_FRAC: usize = 0x3A8;
const ADDR_QSPEED: usize = 0x3BC;
const ADDR_PARALYZED: usize = 0x512;
const ADDR_FIRST_UNWALKABLE: u16 = 0x34A;
const ADDR_SLOT1_TYPE: usize = 0x350;
const PLAY_AREA_TILES: u16 = 0x6530;
const MODE_CAVE_SHORTCUT: u8 = 0x0C;
const ADDR_LINK_TIMER: u16 = 0x28;
const ADDR_TRIGGERED_DOOR_CMD: u16 = 0x54;
const ADDR_CUR_OPENED_DOORS: u16 = 0xEE;
const ADDR_SHUTTER_TOUCHED: u16 = 0x519;

/// Turn frames a plan may take: the longest is a walk out of a doorway's far end onto the room's edge (32 pixels).
pub const MAX_TURN_FRAMES: u32 = 32;

// Game direction bits (asm: 1 right, 2 left, 4 down, 8 up).
const RIGHT: u8 = 1;
const LEFT: u8 = 2;
const DOWN: u8 = 4;
const UP: u8 = 8;

/// asm:Link_BorderBounds: outer overworld D, U, R, L; outer dungeon D, U, R, L; inner D, U, R, L.
const BORDER_BOUNDS: [u8; 12] = [0xD6, 0x45, 0xE9, 0x07, 0xC6, 0x55, 0xD9, 0x17, 0xBE, 0x54, 0xD1, 0x1F];
/// asm:Link_ScreenEdgeBounds, by reverse direction index (up, down, left, right).
const SCREEN_EDGES: [u8; 4] = [0x3D, 0xDD, 0x00, 0xF0];
/// asm:Doorway_RequiredCoords, asm:Doorway_BoundsMinOver and asm:Doorway_BoundsMaxOver, asm:Doorway_BoundsMinUnder and
/// asm:Doorway_BoundsMaxUnder (up, down, left, right).
const DOORWAY_REQUIRED: [u8; 4] = [0x78, 0x78, 0x8D, 0x8D];
const DOORWAY_WIDE: [(u8, u8); 4] = [(0x3D, 0x5E), (0xBD, 0xDE), (0x00, 0x21), (0xCF, 0xF1)];
const DOORWAY_NARROW: [(u8, u8); 4] = [(0x3D, 0x5C), (0xBF, 0xDE), (0x00, 0x1F), (0xD2, 0xF1)];
/// The direction bit of each reverse direction index (asm:Dir_ReverseIndexToDir).
const REVERSE_INDEX_DIRS: [u8; 4] = [UP, DOWN, LEFT, RIGHT];
/// asm:Tile_WalkableTilesOW: overworld tiles at or above the limit that the lookup reports as walkable tile $26.
const OW_WALKABLE_TILES: [u8; 9] = [0x8D, 0x91, 0x9C, 0xAC, 0xAD, 0xCC, 0xD2, 0xD5, 0xDF];

/// Link's input state: the RAM his input handling reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub x: u8,
    pub y: u8,
    /// Facing (`ObjDir`, game direction bits).
    pub dir: u8,
    pub offset: i8,
    pub frac: u8,
    pub qspeed: u8,
    pub doorway: u8,
    pub go_straight: bool,
    pub paralyzed: bool,
    pub shoved: bool,
    pub ladder: u8,
    pub state: u8,
    pub level: u8,
    pub mode: u8,
}

impl Link {
    /// From CPU RAM (`$0000-$07FF`).
    pub fn read(ram: &[u8]) -> Link {
        Link {
            x: ram[ADDR_LINK_X],
            y: ram[ADDR_LINK_Y],
            dir: ram[ADDR_LINK_DIR],
            offset: ram[ADDR_GRID_OFFSET] as i8,
            frac: ram[ADDR_POS_FRAC],
            qspeed: ram[ADDR_QSPEED],
            doorway: ram[ADDR_DOORWAY_DIR],
            go_straight: ram[ADDR_GO_STRAIGHT] != 0,
            paralyzed: ram[ADDR_PARALYZED] != 0,
            shoved: ram[ADDR_SHOVE_DIR] != 0,
            ladder: ram[ADDR_LADDER_SLOT],
            state: ram[ADDR_LINK_STATE],
            level: ram[ADDR_LEVEL],
            mode: ram[ADDR_MODE],
        }
    }

    /// In a dungeon room (not a cellar): doorways and room bounds apply.
    fn in_dungeon_room(&self) -> bool {
        self.level != 0 && self.mode != MODE_CELLAR
    }

    fn in_cave(&self) -> bool {
        self.level == 0 && matches!(self.mode, MODE_CAVE | MODE_CAVE_SHORTCUT)
    }
}

/// The next frame of an attack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    /// Press these controller buttons (A or B, with the held direction when it turns Link or aims the boomerang).
    Press(u8),
    /// Hold this direction for one frame: the game walks Link to a grid point or turns him.
    Turn(Direction),
    /// The game can't make this strike from here.
    Impossible,
}

/// The facing an attack in `d` needs: `d`, or N/S for a diagonal boomerang throw (its band test faces that way).
pub fn facing_for(d: Direction) -> Direction {
    match d {
        Direction::NW | Direction::NE => Direction::N,
        Direction::SW | Direction::SE => Direction::S,
        d => d,
    }
}

fn axis(dir: u8) -> u8 {
    if dir & (UP | DOWN) != 0 { UP | DOWN } else if dir & (LEFT | RIGHT) != 0 { LEFT | RIGHT } else { 0 }
}

fn opposite(dir: u8) -> u8 {
    match dir {
        RIGHT => LEFT,
        LEFT => RIGHT,
        DOWN => UP,
        UP => DOWN,
        _ => 0,
    }
}

/// asm:Dir_GetOpposite's index: the lowest set bit (right, left, down, up) as a reverse direction index.
fn reverse_index(bits: u8) -> Option<usize> {
    [(RIGHT, 3), (LEFT, 2), (DOWN, 1), (UP, 0)].into_iter().find(|&(b, _)| bits & b != 0).map(|(_, i)| i)
}

/// Index of a side in [`BORDER_BOUNDS`] (down, up, right, left).
fn border_side(dir: u8) -> usize {
    match dir {
        DOWN => 0,
        UP => 1,
        RIGHT => 2,
        _ => 3,
    }
}

/// Whether a band test reaches its bound: "left or up" tests (`toward` is LEFT or UP) are reached at
/// `coordinate >= bound`, the others at `coordinate < bound` (asm:Link_MaskInputInBorder).
fn band_reached(toward: u8, coordinate: u8, bound: u8) -> bool {
    if toward & (LEFT | UP) != 0 { coordinate >= bound } else { coordinate < bound }
}

/// asm:Link_FilterInput: (inner band reached, outer band reached) for Link facing `facing` at (x, y).
fn border_bands(level: u8, facing: u8, x: u8, y: u8) -> (bool, bool) {
    if !matches!(facing, RIGHT | LEFT | DOWN | UP) {
        return (false, false);
    }
    let coordinate = if facing & (LEFT | RIGHT) != 0 { x } else { y };
    let outer_set = if level == 0 { 0 } else { 4 };
    let ahead = border_side(facing);
    // Test 1: the inner band on the side Link faces ([00] = the opposite of his facing).
    let inner = band_reached(opposite(facing), coordinate, BORDER_BOUNDS[8 + ahead]);
    // Test 2: the outer band on the same side if test 1 hit, else on the side behind him ([00] = his facing).
    let outer = if inner {
        band_reached(opposite(facing), coordinate, BORDER_BOUNDS[outer_set + ahead])
    } else {
        band_reached(facing, coordinate, BORDER_BOUNDS[outer_set + border_side(opposite(facing))])
    };
    (inner, outer)
}

/// Whether a newly pressed A (`b_item` false) or B survives asm:Link_FilterInput for Link facing `facing` at (x, y):
/// the outer band drops both, the inner band drops B.
pub fn button_survives(level: u8, facing: u8, x: u8, y: u8, b_item: bool) -> bool {
    let (inner, outer) = border_bands(level, facing, x, y);
    !outer && !(b_item && inner)
}

/// Facing into the room from wherever Link stands, along his axis (the forced step of a doorway goes that way).
fn faces_into_room(s: &Link) -> bool {
    match s.dir {
        UP => s.y > 0x8D,
        DOWN => s.y < 0x8D,
        LEFT => s.x > 0x78,
        RIGHT => s.x < 0x78,
        _ => false,
    }
}

/// The next input of an attack of `kind` in `d` for Link in state `s` (not halted, in his normal state).
pub fn next(kind: ActionKind, d: Direction, s: &Link) -> Next {
    let b_item = matches!(kind, ActionKind::Bomb | ActionKind::Arrow | ActionKind::Boomerang);
    let facing = s.dir;
    let want = d.value();
    let survives = |f: u8| button_survives(s.level, f, s.x, s.y, b_item);
    // Held by a Like-Like the directions are discarded; knocked back they are read but not turned into a facing.
    let turns = !(s.paralyzed || s.shoved);
    match kind {
        ActionKind::Sword | ActionKind::Beams => {
            let same_axis = axis(facing) == axis(want);
            let in_doorway = s.in_dungeon_room() && s.doorway != 0;
            let turns_now = facing == want || (turns && (same_axis || (s.offset == 0 && !in_doorway)));
            if turns_now && survives(facing) {
                let held = if facing == want { 0 } else { direction_buttons(d) };
                return Next::Press(buttons::A | held);
            }
        }
        ActionKind::Boomerang => {
            if survives(facing) && (!s.paralyzed || want == facing) {
                let held = if s.paralyzed { 0 } else { direction_buttons(d) };
                return Next::Press(buttons::B | held);
            }
        }
        ActionKind::Bomb | ActionKind::Arrow => {
            if facing == want && survives(facing) {
                return Next::Press(buttons::B);
            }
        }
        ActionKind::Move | ActionKind::Wait | ActionKind::Clip => return Next::Impossible,
    }
    let target = facing_for(d);
    if !turns || facing == target.value() {
        return Next::Impossible;
    }
    if s.in_dungeon_room() && s.doorway != 0 && axis(facing) != axis(target.value()) && !faces_into_room(s) {
        return Next::Impossible;
    }
    Next::Turn(target)
}

/// The tile test of Link on a grid point heading `dir` (asm:Tile_GetAheadOfObj, asm:Tile_GetAtObjOffset,
/// asm:Walker_TestTileAhead): the tile map is read as the game reads it, out-of-area rows included. A ladder crossing
/// is not walkable here: that frame the game lays the ladder instead of moving Link.
fn tile_ahead_walkable(view: &GameView, s: &Link, dir: u8) -> bool {
    let (x, hot_y) = (s.x, s.y.wrapping_add(0x0B));
    let (tx, ty) = match dir {
        UP => (x, hot_y.wrapping_sub(8)),
        DOWN if hot_y < 0xDD => (x, hot_y.wrapping_add(8)),
        LEFT if x >= 0x10 => (x.wrapping_sub(8), hot_y),
        RIGHT if x < 0xF0 => (x.wrapping_add(0x10), hot_y),
        _ => (x, hot_y),
    };
    let column = (tx >> 3) as u16;
    let row = (ty.wrapping_sub(0x40) >> 3) as u16;
    let read = |c: u16| view.peek(PLAY_AREA_TILES + c * 22 + row);
    let mut tile = read(column);
    if dir & (UP | DOWN) != 0 {
        tile = tile.max(read(column + 1));
    }
    if s.level == 0 && OW_WALKABLE_TILES.contains(&tile) {
        tile = 0x26;
    }
    // Overworld room $1F: up or down at X $80 above Y $56 is forced walkable (asm:Tile_GetAtObjOffset@SetTile).
    if s.level == 0 && view.location() == 0x1F && dir & (UP | DOWN) != 0 && x == 0x80 && s.y < 0x56 {
        tile = 0x26;
    }
    tile < view.peek(ADDR_FIRST_UNWALKABLE)
}

/// asm:Doorway_CheckLink without the touch: the doorway Link is found in (reverse index), from the frame's start
/// position, facing `dir` and `DoorwayDir` = `doorway`.
fn find_doorway(s: &Link, dir: u8, doorway: u8) -> Option<usize> {
    // [00] is Link's Y facing left or right, else his X; [01] the other.
    let split = |d: u8| if d & (LEFT | RIGHT) != 0 { (s.y, s.x) } else { (s.x, s.y) };
    let search = |(across, along): (u8, u8), spans: &[(u8, u8); 4]| {
        (0..4).rev().find(|&i| across == DOORWAY_REQUIRED[i] && (spans[i].0..spans[i].1).contains(&along))
    };
    if doorway == 0 {
        return search(split(dir), &DOORWAY_WIDE);
    }
    let coords = split(doorway);
    let index = reverse_index(doorway)?;
    let (lo, hi) = DOORWAY_WIDE[index];
    if (lo..hi).contains(&coords.1) && doorway == dir {
        return search(split(dir), &DOORWAY_WIDE);
    }
    search(coords, &DOORWAY_NARROW)
}

/// Whether input `held` (game direction bits) on this frame would take Link out of the room whatever his state:
/// a cave's edge (asm:Warp_CheckCaveEdgeExit tests the input, not the move) or a cellar's stairs (asm:
/// Warp_CheckSubroomExit: Y < $40 with Up held).
fn input_leaves(s: &Link, held: u8) -> bool {
    if s.in_cave()
        && let Some(i) = reverse_index(held)
    {
        let coordinate = if i >= 2 { s.x } else { s.y };
        if coordinate == SCREEN_EDGES[i] {
            return true;
        }
    }
    s.mode == MODE_CELLAR && s.y < 0x40 && held & UP != 0
}

/// Why the buttons can't make a strike from here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A Like-Like holds Link or he is knocked back: no turn.
    Held,
    /// asm:Link_FilterInput drops the button where Link faces the strike's way (or, in a dungeon's outer band, the
    /// turn's input): only walking out of the band first (a MOVE) would do.
    Band,
    /// In a doorway facing out of the room: a sideways press walks Link through the door (asm:Link_ModifyDirInDoorway).
    Doorway,
    /// The input would leave the room: a screen or cave edge, a cellar's stairs, a door it would unlock (a key) or
    /// pass out of the room (a false or opened bomb wall).
    Leaves,
    /// On the laid ladder only its own axis is modelled.
    Ladder,
    /// Not an attack, or longer than `MAX_TURN_FRAMES`.
    Other,
}

/// One frame of Link in his normal state holding direction `t` (asm:Link_HandleInput, asm:Walker_Move,
/// asm:Obj_MoveByQSpeed, asm:Link_EndMoveAndAnimate's grid fix-up): his next state, or why the frame is refused
/// (it would leave the room or open a door, the input is dropped, or the laid ladder).
pub fn step(view: &GameView, s: &Link, t: Direction) -> Result<Link, Refusal> {
    let mut n = *s;
    let mut input = t.value();
    // asm:Link_FilterInput: in a dungeon's outer band the input keeps only the facing's axis.
    if s.level != 0 && border_bands(s.level, s.dir, s.x, s.y).1 {
        input &= axis(s.dir);
    }
    // asm:Link_ModifyDirInDoorway: forward if held, else backward if held, else forward.
    if s.level != 0 && s.doorway != 0 && input != 0 {
        input = if s.dir & input != 0 {
            s.dir
        } else if opposite(s.dir) & input != 0 {
            opposite(s.dir)
        } else {
            s.dir
        };
    }
    if input == 0 {
        return Err(Refusal::Band);
    }
    if input_leaves(s, input) {
        return Err(Refusal::Leaves);
    }
    // The direction rules: on a grid point the held direction becomes the facing; between grid points the six
    // cases (asm:Link_ModifyDirOnGridLine).
    if s.offset == 0 {
        n.go_straight = false;
        n.dir = input;
    } else if input == opposite(s.dir) {
        n.dir = input;
    } else if input != s.dir && !s.go_straight && s.offset.unsigned_abs() < 4 {
        let heading_negative = s.dir & (LEFT | UP) != 0;
        if heading_negative == (s.offset < 0) {
            n.dir = opposite(s.dir);
            n.offset = if s.offset < 0 { 8 + s.offset } else { s.offset - 8 };
        }
    }
    let mut moving = if s.offset == 0 { input } else { n.dir };
    // asm:Walker_Move: a cave's person line; the room bound outside a doorway; the doorways (from the frame's start
    // position, with the facing just set).
    if s.in_cave() && moving == UP && s.y < 0x8E {
        moving = 0;
    }
    if s.in_dungeon_room() {
        let slot1 = view.peek(ADDR_SLOT1_TYPE as u16);
        if moving == UP && s.y < 0x8E && (slot1 == 0x36 || (0x4B..0x53).contains(&slot1)) {
            moving = 0;
        }
        if s.doorway == 0
            && let Some(d) = Direction::from_value(moving)
            && crate::game::at_dungeon_bound(s.x as i32, s.y as i32, d)
        {
            moving = 0;
        }
        match find_doorway(s, n.dir, s.doorway) {
            None => n.doorway = 0,
            Some(i) if input == REVERSE_INDEX_DIRS[i] => match door_touch(view, input) {
                // The door lets him through: he faces and walks into the doorway (asm:Doorway_CheckLink).
                Touch::Passes => {
                    n.dir = input;
                    n.doorway = input;
                    moving = input;
                }
                // It stops him only while he is outside a doorway.
                Touch::Blocks => {
                    if s.doorway == 0 {
                        moving = 0;
                    }
                }
                Touch::Unlocks | Touch::LeavesRoom => return Err(Refusal::Leaves),
            },
            Some(_) => {}
        }
    }
    // asm:Walker_CheckTileCollision: on a grid point outside a doorway, the tile ahead and the screen edge.
    if s.offset == 0 && !(s.in_dungeon_room() && n.doorway != 0) && moving != 0 {
        if s.level == 0
            && s.ladder == 0
            && let Some(i) = reverse_index(input)
            && (if i >= 2 { s.x } else { s.y }) == SCREEN_EDGES[i]
        {
            return Err(Refusal::Leaves);
        }
        if !tile_ahead_walkable(view, s, moving) {
            moving = 0;
        }
    }
    if s.ladder != 0 {
        moving = ladder_moving(view, s, &mut n, input, moving)?;
    }
    if moving != 0 {
        let positive = moving & (RIGHT | DOWN) != 0;
        let horizontal = moving & (RIGHT | LEFT) != 0;
        for _ in 0..4 {
            let (frac, pixel) = if positive {
                n.frac.overflowing_add(n.qspeed)
            } else {
                n.frac.overflowing_sub(n.qspeed)
            };
            n.frac = frac;
            if pixel && n.offset.unsigned_abs() != 8 {
                let delta: i8 = if positive { 1 } else { -1 };
                n.offset += delta;
                if horizontal {
                    n.x = n.x.wrapping_add_signed(delta);
                } else {
                    n.y = n.y.wrapping_add_signed(delta);
                }
            }
        }
    }
    if n.offset != 0 && n.offset % 8 == 0 {
        n.offset = 0;
    }
    if n.offset == 0 {
        n.x &= 0xF8;
        n.y = (n.y & 0xF8) | 5;
    }
    Ok(n)
}

/// asm:Ladder_CheckAndDraw: the moving direction while the ladder lies (it stays while Link keeps to its line within
/// $10 pixels; on it he goes on in its direction or back the way he came, any other input stops him). On the frame it
/// is put away Link doesn't move. Its upward case E (a second tile test) is not modelled.
fn ladder_moving(view: &GameView, s: &Link, n: &mut Link, input: u8, moving: u8) -> Result<u8, Refusal> {
    let slot = s.ladder as u16;
    let (lx, ly) = (view.peek(ADDR_LINK_X as u16 + slot), view.peek(ADDR_LINK_Y as u16 + slot));
    let (ldir, lstate) = (view.peek(ADDR_LINK_DIR as u16 + slot), view.peek(ADDR_LINK_STATE as u16 + slot));
    let distance = if lstate == 0 {
        None
    } else if ldir & (UP | DOWN) != 0 {
        (s.x == lx).then(|| (s.y.wrapping_add(3).wrapping_sub(ly) as i8).unsigned_abs())
    } else {
        (s.y.wrapping_add(3) == ly).then(|| (s.x.wrapping_sub(lx) as i8).unsigned_abs())
    };
    let on = match distance {
        Some(d) if d < 0x10 => true,
        Some(0x10) => n.dir == ldir && lstate == 1,
        _ => false,
    };
    if !on {
        n.ladder = 0;
        return Ok(0);
    }
    let distance = distance.unwrap_or(0);
    Ok(if input == 0 {
        0
    } else if distance != 0 && n.dir == ldir {
        n.dir
    } else if ldir == moving {
        moving
    } else if opposite(ldir) == n.dir {
        n.dir
    } else if ldir == UP && input == UP {
        return Err(Refusal::Ladder);
    } else {
        0
    })
}

/// Whether pressing `press` (controller bits) now would take Link out of the room or through a door: the input
/// alone leaves a cave or cellar, or the held direction touches a door he can pass on the doorway's floor spot
/// (asm:Doorway_CheckLink restores the move the sword stopped, and a false or bombed wall changes rooms at once).
fn press_leaves(view: &GameView, s: &Link, press: u8) -> bool {
    let held = [(buttons::RIGHT, RIGHT), (buttons::LEFT, LEFT), (buttons::DOWN, DOWN), (buttons::UP, UP)]
        .into_iter()
        .filter(|&(b, _)| press & b != 0)
        .fold(0, |bits, (_, g)| bits | g);
    if held == 0 {
        return false;
    }
    if input_leaves(s, held) {
        return true;
    }
    if !s.in_dungeon_room() || !matches!(held, RIGHT | LEFT | DOWN | UP) {
        return false;
    }
    // The facing after the press frame's direction rules (a grid point, or a turn back between grid points).
    let facing = if s.offset == 0 || held == opposite(s.dir) { held } else { s.dir };
    let Some(i) = find_doorway(s, facing, s.doorway) else { return false };
    if held != REVERSE_INDEX_DIRS[i] {
        return false;
    }
    match door_touch(view, held) {
        Touch::Blocks => false,
        Touch::Unlocks | Touch::LeavesRoom => true,
        // He walks a pixel or two into the doorway (two frames on a lag frame): out of the room only at its edge.
        Touch::Passes => {
            let coordinate = if i >= 2 { s.x } else { s.y };
            coordinate.abs_diff(SCREEN_EDGES[i]) <= 4
        }
    }
}

/// What touching the door on side `dir` (game direction bit) does (asm:Door_Touch_JumpTable): an open doorway, an
/// open locked door and an open shutter let Link into the doorway; a locked door he has a key for opens (a key
/// spent); a false wall at the end of its push timer and an opened bomb wall change rooms at once; the rest stop
/// him.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Touch {
    Blocks,
    Passes,
    Unlocks,
    LeavesRoom,
}

fn door_touch(view: &GameView, dir: u8) -> Touch {
    use crate::room_state::DoorType;
    let Some(d) = Direction::from_value(dir) else { return Touch::Blocks };
    let Some(door) = crate::room_state::decode_room_state(view.ram()).door(d) else { return Touch::Blocks };
    let opened = view.peek(ADDR_CUR_OPENED_DOORS) & dir != 0;
    let command = view.peek(ADDR_TRIGGERED_DOOR_CMD) != 0;
    // Link's timer as the handler reads it: the NMI has counted it down once this frame.
    let timer = view.peek(ADDR_LINK_TIMER).saturating_sub(1);
    let unless_timer = if timer != 0 { Touch::Blocks } else { Touch::Passes };
    match door.door_type {
        DoorType::Open => Touch::Passes,
        DoorType::Wall => Touch::Blocks,
        DoorType::FalseWall | DoorType::FalseWall2 if timer == 1 => Touch::LeavesRoom,
        DoorType::FalseWall | DoorType::FalseWall2 => Touch::Blocks,
        DoorType::Bomb if opened => Touch::LeavesRoom,
        DoorType::Bomb => Touch::Blocks,
        DoorType::Key | DoorType::Key2 if opened => Touch::Passes,
        // A door command running lets Link through a closed locked door once his timer is out (to the next room).
        DoorType::Key | DoorType::Key2 if command => {
            if timer != 0 { Touch::Blocks } else { Touch::LeavesRoom }
        }
        DoorType::Key | DoorType::Key2 if view.keys() > 0 || view.magic_key() => Touch::Unlocks,
        DoorType::Key | DoorType::Key2 => Touch::Blocks,
        DoorType::Shutter if command || !opened => Touch::Blocks,
        DoorType::Shutter if view.peek(ADDR_SHUTTER_TOUCHED) & dir != 0 => unless_timer,
        DoorType::Shutter => Touch::Passes,
    }
}

/// The turn frames an attack of `kind` in `d` takes from `view` (0: the press is the next frame), or `None` when
/// the game can't make it by buttons from here (or only by leaving the room or opening a door): [`explain`] says why.
/// Weapon readiness and inventory are the mask's (`GameView::weapon_ready`, `available_actions`).
pub fn plan(view: &GameView, kind: ActionKind, d: Direction) -> Option<u32> {
    explain(view, kind, d).ok()
}

/// [`plan`] with the reason of a refusal.
pub fn explain(view: &GameView, kind: ActionKind, d: Direction) -> Result<u32, Refusal> {
    simulate(view, kind, d).map(|(frames, _)| frames)
}

/// Link (the input model) on the frame an attack of `kind` in `d` presses its button, after [`plan`]'s turn frames;
/// None where [`plan`] refuses.  A bomb or an arrow starts from his position then (asm:Link_PlaceWeaponAhead).
pub fn press_link(view: &GameView, kind: ActionKind, d: Direction) -> Option<Link> {
    simulate(view, kind, d).ok().map(|(_, s)| s)
}

fn simulate(view: &GameView, kind: ActionKind, d: Direction) -> Result<(u32, Link), Refusal> {
    let mut s = Link::read(&view.ram()[..0x800]);
    for frames in 0..=MAX_TURN_FRAMES {
        match next(kind, d, &s) {
            Next::Press(press) if press_leaves(view, &s, press) => return Err(Refusal::Leaves),
            Next::Press(_) => return Ok((frames, s)),
            Next::Impossible => {
                let target = facing_for(d).value();
                return Err(if matches!(kind, ActionKind::Move | ActionKind::Wait | ActionKind::Clip) {
                    Refusal::Other
                } else if s.paralyzed || s.shoved {
                    Refusal::Held
                } else if s.dir == target {
                    Refusal::Band
                } else {
                    Refusal::Doorway
                });
            }
            Next::Turn(t) => s = step(view, &s, t)?,
        }
    }
    Err(Refusal::Other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn border_bands_follow_the_table() {
        // Dungeon, facing up: A dies at Y < $55 (outer), B at Y < $54 (inner) too; Y $54 keeps both (test 1 misses
        // at $54, so the outer band is tested behind him).
        assert!(!button_survives(1, UP, 0x78, 0x50, false));
        assert!(button_survives(1, UP, 0x78, 0x54, false) && button_survives(1, UP, 0x78, 0x54, true));
        assert!(button_survives(1, UP, 0x78, 0x55, true));
        // Facing right: inner X >= $D1 drops B only, outer X >= $D9 (dungeon) drops A.
        assert!(button_survives(1, RIGHT, 0xD0, 0x8D, true));
        assert!(button_survives(1, RIGHT, 0xD1, 0x8D, false) && !button_survives(1, RIGHT, 0xD1, 0x8D, true));
        assert!(!button_survives(1, RIGHT, 0xD9, 0x8D, false));
        // Behind: facing down just inside the top outer band (overworld Y < $45) drops both.
        assert!(!button_survives(0, DOWN, 0x78, 0x44, false));
        assert!(button_survives(0, DOWN, 0x78, 0x45, true));
    }

    #[test]
    fn the_sword_turns_on_its_press_and_items_turn_first() {
        let s = Link {
            x: 0x78,
            y: 0x8D,
            dir: UP,
            offset: 0,
            frac: 0,
            qspeed: 0x60,
            doorway: 0,
            go_straight: false,
            paralyzed: false,
            shoved: false,
            ladder: 0,
            state: 0,
            level: 1,
            mode: 5,
        };
        assert_eq!(next(ActionKind::Sword, Direction::E, &s), Next::Press(buttons::A | buttons::RIGHT));
        assert_eq!(next(ActionKind::Sword, Direction::N, &s), Next::Press(buttons::A));
        assert_eq!(next(ActionKind::Boomerang, Direction::SW, &s), Next::Press(buttons::B | buttons::DOWN | buttons::LEFT));
        assert_eq!(next(ActionKind::Arrow, Direction::E, &s), Next::Turn(Direction::E));
        assert_eq!(next(ActionKind::Bomb, Direction::N, &s), Next::Press(buttons::B));
        // Off the grid across the facing: hold the direction until the game reaches a grid point.
        let off = Link { offset: -3, y: 0x8A, ..s };
        assert_eq!(next(ActionKind::Sword, Direction::E, &off), Next::Turn(Direction::E));
        assert_eq!(next(ActionKind::Sword, Direction::S, &off), Next::Press(buttons::A | buttons::DOWN));
        // Held by a Like-Like: the facing only.
        let held = Link { paralyzed: true, ..s };
        assert_eq!(next(ActionKind::Sword, Direction::E, &held), Next::Impossible);
        assert_eq!(next(ActionKind::Boomerang, Direction::N, &held), Next::Press(buttons::B));
        // In a doorway facing out of the room a sideways turn walks him through the door: impossible.
        let doorway = Link { y: 0x55, doorway: UP, ..s };
        assert_eq!(next(ActionKind::Sword, Direction::E, &doorway), Next::Impossible);
        let leaving = Link { y: 0x55, doorway: UP, dir: DOWN, ..s };
        assert_eq!(next(ActionKind::Sword, Direction::E, &leaving), Next::Turn(Direction::E));
    }
}
