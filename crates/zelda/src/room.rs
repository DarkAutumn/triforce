//! Room walkability, exits and the wavefront.
//!
//! Walkability uses the NES threshold system from asm:Tile_GetAheadOfObj.

use crate::enums::{Direction, MapLocation, Tile};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::sync::OnceLock;

pub const COLS: i32 = 32;
pub const ROWS: i32 = 22;
const CELLS: usize = (COLS * ROWS) as usize;

pub const NORTH_DOOR_TILE: Tile = (0xF, 0x2);
pub const WEST_DOOR_TILE: Tile = (0x2, 0xA);
pub const EAST_DOOR_TILE: Tile = (0x1C, 0xA);
pub const SOUTH_DOOR_TILE: Tile = (0xF, 0x12);

pub const TOP_CAVE_TILE: u8 = 0xF3;
pub const BOTTOM_CAVE_TILE: u8 = 0x24;
/// The top-left tile of an underworld stairs square ($70 $72 over $71 $73).
pub const STAIRS_TOP_LEFT_TILE: u8 = 0x70;

/// Tiles with value < threshold are walkable (asm:Room_ObjBoundsOW, asm:Room_ObjBoundsUW).
pub const OW_FIRST_UNWALKABLE: u8 = 0x89;
pub const UW_FIRST_UNWALKABLE: u8 = 0x78;

/// Overworld asm:Tile_WalkableTilesOW substitutions.
pub const OW_WALKABLE_OVERRIDES: [u8; 9] = [0x8D, 0x91, 0x9C, 0xAC, 0xAD, 0xCC, 0xD2, 0xD5, 0xDF];

/// Cave entry tiles the agent may walk into (asm:Warp_CheckUnderLink starts the cave from the tile under Link).
const OW_CAVE_ENTRY_TILES: [u8; 1] = [TOP_CAVE_TILE];

/// Underworld water (asm:RoomLayout_LayOutUWFloor's `W` square): a wall, but for the ladder.
pub const WATER_TILE: u8 = 0xF4;
/// Overworld water as the ladder sees it: tiles `$8D`-`$98` the lookup returns, which has already made the walkable
/// `$8D` and `$91` tile `$26` (asm:Link_EndMoveAndAnimate@CheckWaterOW, asm:Tile_WalkableTilesOW).
pub const OW_WATER_TILES: std::ops::RangeInclusive<u8> = 0x8D..=0x98;
/// The only overworld rooms where the ladder is laid (asm:Ladder_RoomIdsOW).
pub const OW_LADDER_ROOMS: [u8; 6] = [0x17, 0x18, 0x19, 0x27, 0x4F, 0x5F];
/// Overworld room `$1F`: for Link a vertical test at X `$80` with Y < `$56` reads walkable, the gap in its N wall
/// into `$0F` (asm:Tile_GetAtObjOffset@SetTile).
const GAP_ROOM: u8 = 0x1F;
const GAP_X: i32 = 0x80;
const GAP_BELOW_Y: i32 = 0x56;
/// `Room::ladder_axes` bits: a water tile the ladder crosses vertically / horizontally.
pub const LADDER_V: u8 = 1;
pub const LADDER_H: u8 = 2;

/// Link holding the ladder can lay it in `location`: every dungeon room but a cellar, the overworld's ladder rooms
/// (asm:Link_EndMoveAndAnimate@CheckLadderRoom).
pub fn lays_ladder_in(location: MapLocation) -> bool {
    !location.in_cave && (location.level != 0 || OW_LADDER_ROOMS.contains(&location.value))
}

/// Locked door tiles $98-$A7 (N, S, W, E).
pub fn is_door_tile(tile: u8) -> bool {
    (0x98..0xA8).contains(&tile)
}

/// Barred (shutter) door tiles $A8-$AF.
pub fn is_barred_tile(tile: u8) -> bool {
    (0xA8..0xB0).contains(&tile)
}

fn ow_override(tile: u8) -> bool {
    OW_WALKABLE_OVERRIDES.contains(&tile)
}

const fn door_tile(direction: Direction) -> Tile {
    match direction {
        Direction::N => NORTH_DOOR_TILE,
        Direction::S => SOUTH_DOOR_TILE,
        Direction::E => EAST_DOOR_TILE,
        _ => WEST_DOOR_TILE,
    }
}

const fn offset(direction: Direction) -> Tile {
    match direction {
        Direction::N => (0, -1),
        Direction::S => (0, 1),
        Direction::E => (1, 0),
        Direction::W => (-1, 0),
        _ => (0, 0),
    }
}

const fn exit_slot(direction: Direction) -> usize {
    match direction {
        Direction::N => 0,
        Direction::S => 1,
        Direction::E => 2,
        _ => 3,
    }
}

const EXIT_ORDER: [Direction; 4] = [Direction::N, Direction::S, Direction::E, Direction::W];

/// The 32x22 PlayAreaTiles, column-major like RAM: `tiles[x * 22 + y]`.
pub type Tiles = [u8; CELLS];

/// A wavefront target: a tile, or every exit tile of a direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Target {
    Tile(Tile),
    Dir(Direction),
}

/// Room exits: per direction the exit tiles (`None` when the room has no such exit key), plus the
/// tile -> direction lookup of the original exits.
#[derive(Clone, Debug, Default)]
pub struct Exits {
    lists: [Option<Vec<Tile>>; 4],
    by_tile: Vec<(Tile, Direction)>,
}

impl Exits {
    /// The exit tiles in `direction`, None if the room has no such exit.
    pub fn get(&self, direction: Direction) -> Option<&[Tile]> {
        if !direction.is_cardinal() {
            return None;
        }
        self.lists[exit_slot(direction)].as_deref()
    }

    /// Whether `direction` is an exit key (possibly with an empty tile list, on the overworld).
    pub fn contains(&self, direction: Direction) -> bool {
        self.get(direction).is_some()
    }

    /// Directions with an exit key, in N, S, E, W order.
    pub fn directions(&self) -> impl Iterator<Item = Direction> + '_ {
        EXIT_ORDER.into_iter().filter(|d| self.contains(*d))
    }

    /// The exit direction of an exit tile (the last assignment wins, as in the Python dict).
    pub fn direction_of(&self, tile: Tile) -> Option<Direction> {
        self.by_tile
            .iter()
            .rev()
            .find(|(t, _)| *t == tile)
            .map(|(_, d)| *d)
    }

    fn set(&mut self, direction: Direction, tiles: Vec<Tile>, map_tiles: bool) {
        if map_tiles {
            for &t in &tiles {
                self.by_tile.push((t, direction));
            }
        }
        self.lists[exit_slot(direction)] = Some(tiles);
    }
}

/// A jump of the path search: Link at rest on tile `from` reaches tile `to` by one action heading `direction` (a
/// grid-point skip, `crate::clips::skip_jumps`), worth `cost` tiles of walking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Jump {
    pub from: Tile,
    pub to: Tile,
    pub direction: Direction,
    pub cost: i32,
}

/// A room's walkability model.
#[derive(Clone, Debug)]
pub struct Room {
    pub location: MapLocation,
    tiles: Tiles,
    overworld: bool,
    threshold: u8,
    corridor: [bool; CELLS],
    loaded: bool,
    /// Link holds the ladder (`Room::new`); in an overworld room only where the game lays it ([`lays_ladder_in`]).
    ladder: bool,
    /// Per tile, the axes (`LADDER_V`, `LADDER_H`) along which Link holding the ladder crosses this water tile.
    ladder_axes: [u8; CELLS],
    /// Overworld room `$1F`'s gap at X `$80` (`GAP_ROOM`).
    link_gap: bool,
    /// The room's skips as path-search jumps, built on first use (`crate::clips::skip_jumps`).
    jumps: OnceLock<Vec<Jump>>,
    /// Link's tile rows (game.yaml `span`) of the room's W wrap exit (`crate::game_map::RoomExit::wrap`), left only
    /// by `CLIP E` ([`Room::with_wrap_exit`]).
    wrap_exit: Option<(i32, i32)>,
    /// The tile rows (Link's, game.yaml `wrap_in`) on which a screen wrap reaches the side of the room an exit is on,
    /// and that exit's direction ([`Room::with_wrap_in`]).
    wrap_in: Option<(Direction, (i32, i32))>,
    /// Where Link stands to sail from the room's raft dock (game.yaml `raft_exits` `dock`, in pixels: X, Y), round 36:
    /// the dock warp's water spot is the dock's row W of it ([`Room::with_dock`]).
    dock: Option<(i32, i32)>,
    /// Per tile, the extra cost of a path-search step onto it (game.yaml `path_costs`, [`Room::with_path_costs`]);
    /// empty: none.
    path_cost: Vec<u8>,
    pub exits: Exits,
    pub cave_tile: Option<Tile>,
}

fn index(tc: i32, tr: i32) -> usize {
    (tc * ROWS + tr) as usize
}

fn in_bounds(tc: i32, tr: i32) -> bool {
    (0..COLS).contains(&tc) && (0..ROWS).contains(&tr)
}

impl Room {
    /// `ladder`: Link holds the ladder (`GameView::ladder`); in a room where the game lays it ([`lays_ladder_in`])
    /// its crossings become passable.
    pub fn new(location: MapLocation, tiles: &Tiles, ladder: bool) -> Room {
        let overworld = location.level == 0;
        let threshold = if overworld {
            OW_FIRST_UNWALKABLE
        } else {
            UW_FIRST_UNWALKABLE
        };
        let mut room = Room {
            location,
            tiles: *tiles,
            overworld,
            threshold,
            corridor: [false; CELLS],
            loaded: false,
            ladder: ladder && (!overworld || lays_ladder_in(location)),
            ladder_axes: [0; CELLS],
            link_gap: location == MapLocation::new(0, GAP_ROOM, false),
            jumps: OnceLock::new(),
            exits: Exits::default(),
            cave_tile: None,
            wrap_exit: None,
            wrap_in: None,
            dock: None,
            path_cost: Vec::new(),
        };
        room.loaded = tiles
            .iter()
            .any(|&t| t < threshold || (overworld && ow_override(t)));
        if !overworld && !location.in_cave {
            room.compute_corridor_tiles();
        }
        if room.ladder && lays_ladder_in(location) {
            room.compute_ladder_crossings();
        }
        room.exits = room.exit_tiles();
        // Underworld rooms have no cave mouths ($F3 is a cellar's background).
        room.cave_tile = overworld
            .then(|| {
                (0..COLS)
                    .flat_map(|x| (0..ROWS).map(move |y| (x, y)))
                    .find(|&(x, y)| tiles[index(x, y)] == TOP_CAVE_TILE)
            })
            .flatten();
        room
    }

    /// The top-left tiles of the room's stairs squares (tiles $70-$73; asm:Warp_CheckUnderLink sends Link down from a
    /// square-aligned stand on one): where Link enters an underworld cellar.
    pub fn stairs_tiles(&self) -> Vec<Tile> {
        (0..COLS)
            .flat_map(|x| (0..ROWS).map(move |y| (x, y)))
            .filter(|&(x, y)| self.tiles[index(x, y)] == STAIRS_TOP_LEFT_TILE)
            .collect()
    }

    pub fn threshold(&self) -> u8 {
        self.threshold
    }

    /// The game's own limit (`RoomFirstUnwalkableTile` `$34A`) where it is above the overworld default (round 26):
    /// the drained pond of Q1 room 0x42, whose water asm:FluteSecret_CuePondPaletteRow makes walkable by raising the
    /// limit to `$99` (its stairs lie in that water). Every start state holds the defaults (`$89` overworld, `$78`
    /// underworld), so no other room changes.
    pub fn with_first_unwalkable(mut self, limit: u8) -> Room {
        if !self.overworld || limit <= self.threshold {
            return self;
        }
        self.threshold = limit;
        self.exits = self.exit_tiles();
        self
    }

    /// Opened bombable walls (round 26, 7:0x69 N): the game lets Link through an opened bomb hole by its door type
    /// (asm:Doorway_CheckLink, [`crate::GameView::is_bomb_door_open`]), whatever tile its rim shows (the N and W rims
    /// are unwalkable tiles), so the hole is a doorway corridor and an exit as an open door is. A dungeon room's
    /// doors only; a door whose tile already reads walkable is left as it is.
    pub fn with_open_doorways(mut self, directions: &[Direction]) -> Room {
        if self.overworld || self.location.in_cave {
            return self;
        }
        for &direction in directions {
            if self.is_door_tile_walkable(direction) {
                continue;
            }
            self.add_locked_door_corridor(direction);
            self.exits.set(direction, vec![Self::door_exit_tile(direction)], true);
        }
        self
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Link holds the ladder (the water crossings and the ladder clip's jumps).
    pub fn ladder(&self) -> bool {
        self.ladder
    }

    /// The room with its game.yaml W wrap exit's rows (`GameView::room`): the W edge tiles of those rows become W exit
    /// tiles (solid, so walking never reaches them: only a Link standing at X `$00`, past the bound, is on one), and
    /// the wrap exit's jumps lead there ([`Room::skip_jumps`], with clips).
    pub fn with_wrap_exit(mut self, rows: Option<(i32, i32)>) -> Room {
        self.wrap_exit = rows;
        self.jumps = OnceLock::new();
        if let Some((lo, hi)) = rows {
            let mut tiles = self.exits.get(Direction::W).unwrap_or_default().to_vec();
            let edge: Vec<Tile> = (lo..=hi).map(|r| (0, r)).filter(|t| !tiles.contains(t)).collect();
            tiles.extend(edge);
            self.exits.set(Direction::W, tiles, true);
        }
        self
    }

    /// Link's tile rows of the room's W wrap exit, if it has one.
    pub fn wrap_exit(&self) -> Option<(i32, i32)> {
        self.wrap_exit
    }

    /// The room with its game.yaml `wrap_in` exit (`GameView::room`): the exit `direction`'s side of the room is
    /// reached from the other side by the screen wrap toward it on `rows` (0x55's W third, cut off by the river: CLIP
    /// E from the dock side), a jump of the path search with clips ([`Room::skip_jumps`]).
    pub fn with_wrap_in(mut self, wrap_in: Option<(Direction, (i32, i32))>) -> Room {
        self.wrap_in = wrap_in;
        self.jumps = OnceLock::new();
        self
    }

    /// The exit direction and Link's tile rows of the room's `wrap_in` exit, if it has one.
    pub fn wrap_in(&self) -> Option<(Direction, (i32, i32))> {
        self.wrap_in
    }

    /// The room with its raft dock (`GameView::room`, game.yaml `raft_exits` `dock: [col, row]`): Link sails from
    /// (col * 8, row * 8 + `$3D`), the dock's trigger (asm:Dock_Update tests Link's X and Y only).
    pub fn with_dock(mut self, dock: Option<Tile>) -> Room {
        self.dock = dock.map(|(c, r)| (c * 8, r * 8 + 0x3D));
        self
    }

    /// Link's position (X, Y) on the room's raft dock trigger, if it has one.
    pub fn dock(&self) -> Option<(i32, i32)> {
        self.dock
    }

    /// The room with its game.yaml `path_costs` (`GameView::room`): every step of the path search onto a tile of one
    /// of `costs`' rectangles costs `1 + cost` tiles instead of 1, so the objective's way (PBRS, the `target`
    /// observation, linger) goes round them when that is shorter (Dungeon 5's 0x37: the X `$70` column, one ladder
    /// crossing, instead of the straight `$78` one; docs/routes/d5.md).  Walkability and the action mask are
    /// unchanged, and every tile keeps a distance.
    pub fn with_path_costs(mut self, costs: &[crate::game_map::PathCost]) -> Room {
        if costs.is_empty() {
            return self;
        }
        let mut grid = vec![0u8; CELLS];
        for c in costs {
            for tc in c.cols.0.max(0)..=c.cols.1.min(COLS - 1) {
                for tr in c.rows.0.max(0)..=c.rows.1.min(ROWS - 1) {
                    let cell = &mut grid[index(tc, tr)];
                    *cell = cell.saturating_add(c.cost);
                }
            }
        }
        self.path_cost = grid;
        self
    }

    /// The extra cost of a path-search step onto tile (column, row): 0 off screen and without `path_costs`.
    pub fn path_cost(&self, tc: i32, tr: i32) -> i32 {
        if self.path_cost.is_empty() || !in_bounds(tc, tr) {
            return 0;
        }
        i32::from(self.path_cost[index(tc, tr)])
    }

    /// Water the ladder is laid over: `$F4` in a dungeon, [`OW_WATER_TILES`] (bar the walkable ones) overworld.
    pub fn is_water(&self, tile: u8) -> bool {
        if self.overworld { OW_WATER_TILES.contains(&tile) && !ow_override(tile) } else { tile == WATER_TILE }
    }

    /// The room's grid-point skips as path-search jumps (`crate::clips::skip_jumps`), built once per room model.
    pub fn skip_jumps(&self) -> &[Jump] {
        self.jumps.get_or_init(|| crate::clips::skip_jumps(self))
    }

    /// Raw tile value (column, row); panics out of bounds.
    pub fn tile(&self, tc: i32, tr: i32) -> u8 {
        self.tiles[index(tc, tr)]
    }

    pub fn tiles(&self) -> &Tiles {
        &self.tiles
    }

    /// NES walkability of a tile; off-screen is walkable (a room transition), as are doorway corridors.  The
    /// terrain test: ladder crossings are not walkable here (`passable` adds them).
    pub fn is_tile_walkable(&self, tc: i32, tr: i32) -> bool {
        if !in_bounds(tc, tr) {
            return true;
        }
        if self.corridor[index(tc, tr)] {
            return true;
        }
        let value = self.tile(tc, tr);
        if self.overworld && ow_override(value) {
            return true;
        }
        value < self.threshold
    }

    /// Walkable, or a ladder crossing along `axis` (`LADDER_V` or `LADDER_H`).
    fn passable(&self, tc: i32, tr: i32, axis: u8) -> bool {
        self.is_tile_walkable(tc, tr) || (in_bounds(tc, tr) && self.ladder_axes[index(tc, tr)] & axis != 0)
    }

    /// Marks the water the ladder crosses (asm:Link_EndMoveAndAnimate@CheckLadder lays it on the one water square
    /// ahead; asm:Ladder_CheckAndDraw lets Link go on only along it and only onto a walkable tile): a run of water
    /// tiles exactly one square (2 tiles) long in a column (row) with a walkable tile at both ends is crossed
    /// vertically (horizontally).  A strip two squares wide can't be crossed.
    fn compute_ladder_crossings(&mut self) {
        let water = |room: &Room, c: i32, r: i32| in_bounds(c, r) && room.is_water(room.tile(c, r));
        let mut marks = Vec::new();
        for (axis, (dc, dr), lines, length) in [(LADDER_V, (0, 1), COLS, ROWS), (LADDER_H, (1, 0), ROWS, COLS)] {
            for line in 0..lines {
                let at = |k: i32| if axis == LADDER_V { (line, k) } else { (k, line) };
                let mut k = 0;
                while k < length {
                    let (c, r) = at(k);
                    if !water(self, c, r) {
                        k += 1;
                        continue;
                    }
                    let start = k;
                    while k < length && water(self, at(k).0, at(k).1) {
                        k += 1;
                    }
                    let (before, after) = (at(start - 1), at(k));
                    if k - start == 2
                        && in_bounds(before.0, before.1)
                        && in_bounds(after.0, after.1)
                        && self.is_tile_walkable(before.0, before.1)
                        && self.is_tile_walkable(after.0, after.1)
                    {
                        marks.push((c, r, axis));
                        marks.push((c + dc, r + dr, axis));
                    }
                }
            }
        }
        for (c, r, axis) in marks {
            self.ladder_axes[index(c, r)] |= axis;
        }
    }

    /// Whether Link at pixel (px, py) can move in `direction` (exact NES hotspots).
    pub fn can_link_move_from(&self, px: i32, py: i32, direction: Direction) -> bool {
        let feet_row = (py - 53).div_euclid(8);
        match direction {
            Direction::W => self.passable((px - 8).div_euclid(8), feet_row, LADDER_H),
            Direction::E => self.passable((px + 16).div_euclid(8), feet_row, LADDER_H),
            Direction::N if self.in_gap(px, py) => true,
            Direction::S if self.in_gap(px, py) => true,
            Direction::N => self.is_vertical_walkable(px.div_euclid(8), (py - 61).div_euclid(8)),
            Direction::S => self.is_vertical_walkable(px.div_euclid(8), (py - 45).div_euclid(8)),
            _ => false,
        }
    }

    /// Link at (px, py) is in overworld room `$1F`'s gap column above Y `$56`, where up and down read walkable.
    fn in_gap(&self, px: i32, py: i32) -> bool {
        self.link_gap && px == GAP_X && py < GAP_BELOW_Y
    }

    fn is_vertical_walkable(&self, col: i32, row: i32) -> bool {
        if !(0..ROWS).contains(&row) {
            return true;
        }
        if !(0..COLS).contains(&col) || col + 1 >= COLS {
            return true;
        }
        if self.corridor[index(col, row)] || self.corridor[index(col + 1, row)] {
            return true;
        }
        let value = self.tile(col, row).max(self.tile(col + 1, row));
        if self.overworld && ow_override(value) {
            return true;
        }
        let passable = |c: i32| self.tile(c, row) < self.threshold || self.ladder_axes[index(c, row)] & LADDER_V != 0;
        passable(col) && passable(col + 1)
    }

    /// The tile(s) the NES collision check reads when Link at (px, py) moves in `direction`.
    pub fn hotspot_tiles(px: i32, py: i32, direction: Direction) -> Vec<Tile> {
        match direction {
            Direction::N => {
                let (row, col) = ((py - 61).div_euclid(8), px.div_euclid(8));
                vec![(col, row), (col + 1, row)]
            }
            Direction::S => {
                let (row, col) = ((py - 45).div_euclid(8), px.div_euclid(8));
                vec![(col, row), (col + 1, row)]
            }
            Direction::W => vec![((px - 8).div_euclid(8), (py - 53).div_euclid(8))],
            Direction::E => vec![((px + 16).div_euclid(8), (py - 53).div_euclid(8))],
            _ => panic!("unsupported direction {direction:?}"),
        }
    }

    fn hotspot_hits(&self, px: i32, py: i32, direction: Direction, values: &[u8]) -> bool {
        Self::hotspot_tiles(px, py, direction)
            .into_iter()
            .any(|(tc, tr)| in_bounds(tc, tr) && values.contains(&self.tile(tc, tr)))
    }

    /// Whether moving in `direction` from (px, py) runs into an overworld cave entrance.
    pub fn is_cave_entry_direction(&self, px: i32, py: i32, direction: Direction) -> bool {
        self.overworld && self.hotspot_hits(px, py, direction, &OW_CAVE_ENTRY_TILES)
    }

    /// Tile-space movement check used by the wavefront (Link's top-left tile at (tc, tr)).
    pub fn can_move(&self, tc: i32, tr: i32, direction: Direction) -> bool {
        match direction {
            Direction::E => tr + 1 < ROWS && self.passable(tc + 2, tr + 1, LADDER_H),
            Direction::W => tr + 1 < ROWS && self.passable(tc - 1, tr + 1, LADDER_H),
            Direction::S => tc + 1 < COLS && (self.tile_in_gap(tc, tr) || self.check_vertical(tc, tr + 2)),
            Direction::N => tc + 1 < COLS && (self.tile_in_gap(tc, tr) || self.check_vertical(tc, tr)),
            _ => false,
        }
    }

    /// [`Room::in_gap`] for Link's top-left tile (column `$10`, rows of Y `$3D`-`$55`).
    fn tile_in_gap(&self, tc: i32, tr: i32) -> bool {
        self.link_gap && tc == GAP_X / 8 && tr <= (GAP_BELOW_Y - 1 - crate::enums::GAMEPLAY_START_Y) / 8
    }

    fn check_vertical(&self, tc: i32, tr: i32) -> bool {
        self.passable(tc, tr, LADDER_V) && self.passable(tc + 1, tr, LADDER_V)
    }

    fn compute_corridor_tiles(&mut self) {
        if self.is_raw_tile_walkable(EAST_DOOR_TILE) {
            let (dc, dr) = EAST_DOOR_TILE;
            for c in dc..COLS {
                self.corridor[index(c, dr)] = true;
                self.corridor[index(c, dr + 1)] = true;
            }
        }
        if self.is_raw_tile_walkable(WEST_DOOR_TILE) {
            let (dc, dr) = WEST_DOOR_TILE;
            for c in 0..=dc {
                self.corridor[index(c, dr)] = true;
                self.corridor[index(c, dr + 1)] = true;
            }
        }
        if self.is_raw_tile_walkable(NORTH_DOOR_TILE) {
            let (dc, dr) = NORTH_DOOR_TILE;
            for r in 0..=dr {
                self.corridor[index(dc, r)] = true;
                self.corridor[index(dc + 1, r)] = true;
            }
        }
        if self.is_raw_tile_walkable(SOUTH_DOOR_TILE) {
            let (dc, dr) = SOUTH_DOOR_TILE;
            for r in dr..ROWS {
                self.corridor[index(dc, r)] = true;
                self.corridor[index(dc + 1, r)] = true;
            }
        }
    }

    fn is_raw_tile_walkable(&self, (tc, tr): Tile) -> bool {
        self.tile(tc, tr) < self.threshold
    }

    fn add_locked_door_corridor(&mut self, direction: Direction) {
        let mut mark = |c: i32, r: i32| self.corridor[index(c, r)] = true;
        match direction {
            Direction::E => {
                let (dc, dr) = EAST_DOOR_TILE;
                for c in dc - 1..COLS {
                    mark(c, dr);
                    mark(c, dr + 1);
                }
            }
            Direction::W => {
                let (dc, dr) = WEST_DOOR_TILE;
                for c in 0..dc + 2 {
                    mark(c, dr);
                    mark(c, dr + 1);
                }
            }
            Direction::N => {
                let (dc, dr) = NORTH_DOOR_TILE;
                for r in 0..dr + 2 {
                    mark(dc, r);
                    mark(dc + 1, r);
                }
            }
            Direction::S => {
                let (dc, dr) = SOUTH_DOOR_TILE;
                for r in dr - 1..ROWS {
                    mark(dc, r);
                    mark(dc + 1, r);
                }
            }
            _ => {}
        }
    }

    fn door_exit_tile(direction: Direction) -> Tile {
        match direction {
            Direction::N => (NORTH_DOOR_TILE.0, 0),
            Direction::S => (SOUTH_DOOR_TILE.0, ROWS - 1),
            Direction::W => (0, WEST_DOOR_TILE.1),
            _ => (COLS - 1, EAST_DOOR_TILE.1),
        }
    }

    /// Whether the raw tile at the door position is walkable (open), regardless of locks/bars.
    pub fn is_door_tile_walkable(&self, direction: Direction) -> bool {
        self.is_raw_tile_walkable(door_tile(direction))
    }

    /// Whether the door in `direction` is a locked door tile in `fresh_tiles`.
    pub fn is_door_locked(direction: Direction, fresh_tiles: &Tiles) -> bool {
        let (c, r) = door_tile(direction);
        is_door_tile(fresh_tiles[index(c, r)])
    }

    /// Whether the door in `direction` is a barred (shutter) tile in `fresh_tiles`.
    pub fn is_door_barred(direction: Direction, fresh_tiles: &Tiles) -> bool {
        let (c, r) = door_tile(direction);
        is_barred_tile(fresh_tiles[index(c, r)])
    }

    /// Whether the wall in `direction` is solid (not a door): a possible bomb wall still intact.
    pub fn is_wall_intact(&self, direction: Direction, fresh_tiles: &Tiles) -> bool {
        let (c, r) = door_tile(direction);
        let tile = fresh_tiles[index(c, r)];
        if is_door_tile(tile) || is_barred_tile(tile) {
            return false;
        }
        tile >= self.threshold
    }

    fn exit_tiles(&self) -> Exits {
        let mut exits = Exits::default();
        if self.location.level == 0 {
            let row = |y: i32| -> Vec<Tile> {
                (0..COLS - 1)
                    .map(|x| (x, y))
                    .filter(|&(x, y)| self.is_tile_walkable(x, y + 1))
                    .collect()
            };
            let col = |x: i32| -> Vec<Tile> {
                (0..ROWS - 1)
                    .map(|y| (x, y))
                    .filter(|&(x, y)| self.is_tile_walkable(x, y + 1))
                    .collect()
            };
            let mut north = row(0);
            // Room $1F's gap: Link walks through its N wall at X $80.
            if self.link_gap && !north.contains(&(GAP_X / 8, 0)) {
                north.push((GAP_X / 8, 0));
            }
            exits.set(Direction::N, north, true);
            exits.set(Direction::S, row(ROWS - 2), true);
            exits.set(Direction::E, col(COLS - 1), true);
            exits.set(Direction::W, col(0), true);
        } else if self.location.in_cave {
            // An item cellar: up the ladder out of the top row (asm:Warp_CheckSubroomExit: Y < $40 moving up).
            let top: Vec<Tile> = (0..COLS - 1)
                .map(|x| (x, 0))
                .filter(|&(x, y)| self.is_tile_walkable(x, y) && self.is_tile_walkable(x + 1, y))
                .collect();
            exits.set(Direction::N, top, true);
        } else {
            for direction in [Direction::N, Direction::S, Direction::W, Direction::E] {
                let (c, r) = door_tile(direction);
                if self.is_tile_walkable(c, r) {
                    exits.set(direction, vec![Self::door_exit_tile(direction)], true);
                }
            }
        }
        exits
    }

    /// The wavefront toward `targets` (`calculate_wavefront_for_link`).  `locked_doors` are treated as open.  With
    /// `clips` the room's grid-point skips are jumps of the search ([`Room::skip_jumps`]; a specialist's `clips:
    /// true`), so a place only a skip reaches (level 4 room `$21`'s island, the W exit of a room with a wrap exit)
    /// still leads to the targets.
    pub fn wavefront(
        &self,
        targets: &[Target],
        impassible: &[Tile],
        locked_doors: &[Direction],
        clips: bool,
    ) -> Wavefront {
        let jumps = if clips { self.skip_jumps() } else { &[] };
        if locked_doors.is_empty() {
            return self.wavefront_inner(targets, impassible, jumps);
        }
        let mut room = self.clone();
        for &direction in locked_doors {
            room.add_locked_door_corridor(direction);
            room.exits.lists[exit_slot(direction)] = Some(vec![Self::door_exit_tile(direction)]);
        }
        room.wavefront_inner(targets, impassible, jumps)
    }

    fn wavefront_inner(&self, targets: &[Target], impassible: &[Tile], jumps: &[Jump]) -> Wavefront {
        let impassible: HashSet<Tile> = impassible.iter().copied().collect();
        let mut starts: HashSet<Tile> = HashSet::new();
        let mut groups: Vec<(Direction, Vec<Tile>)> = Vec::new();
        let mut group = |direction: Direction, tiles: &[Tile]| match groups
            .iter_mut()
            .find(|(d, _)| *d == direction)
        {
            Some((_, list)) => list.extend_from_slice(tiles),
            None => groups.push((direction, tiles.to_vec())),
        };
        for target in targets {
            match *target {
                Target::Dir(direction) => {
                    if let Some(tiles) = self.exits.get(direction) {
                        group(direction, tiles);
                    }
                }
                Target::Tile(tile) => match self.exits.direction_of(tile) {
                    Some(direction) => group(direction, &[tile]),
                    None => {
                        starts.insert(tile);
                    }
                },
            }
        }
        for (direction, tiles) in &groups {
            let (dx, dy) = offset(*direction);
            for &(x, y) in tiles {
                starts.insert((x + dx, y + dy));
            }
        }
        starts.retain(|t| !impassible.contains(t));
        Wavefront::compute(self, starts, &impassible, jumps)
    }
}

/// Neighbor deltas and the direction Link at the neighbor moves to reach the source tile.
const NEIGHBORS: [((i32, i32), Direction); 4] = [
    ((-1, 0), Direction::E),
    ((1, 0), Direction::W),
    ((0, -1), Direction::S),
    ((0, 1), Direction::N),
];

/// Tile distances to a set of targets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wavefront {
    grid: Vec<i32>,
    offscreen: Vec<Tile>,
    /// The skips the search used (`Room::wavefront` with clips), for walking down it (`Wavefront::jumps_from`).
    jumps: Vec<Jump>,
}

impl Wavefront {
    fn compute(room: &Room, starts: HashSet<Tile>, impassible: &HashSet<Tile>, jumps: &[Jump]) -> Wavefront {
        let mut grid = vec![-1i32; CELLS];
        let mut offscreen = Vec::new();
        let mut starts: Vec<Tile> = starts.into_iter().collect();
        starts.sort_unstable();
        for &tile in &starts {
            if in_bounds(tile.0, tile.1) {
                grid[index(tile.0, tile.1)] = 0;
            } else {
                offscreen.push(tile);
            }
        }
        // The walking neighbours that reach `(tx, ty)` (from off screen, every in-bounds neighbour).
        let neighbours = |grid: &[i32], (tx, ty): Tile| -> [Option<Tile>; 4] {
            let source_offscreen = !in_bounds(tx, ty);
            NEIGHBORS.map(|((dx, dy), direction)| {
                let (nx, ny) = (tx + dx, ty + dy);
                let open = in_bounds(nx, ny)
                    && grid[index(nx, ny)] < 0
                    && !impassible.contains(&(nx, ny))
                    && (source_offscreen || room.can_move(nx, ny, direction));
                open.then_some((nx, ny))
            })
        };
        if jumps.is_empty() && room.path_cost.is_empty() {
            // Unit steps: breadth first.
            let mut todo: VecDeque<(Tile, i32)> = starts.into_iter().map(|t| (t, 0)).collect();
            while let Some((tile, dist)) = todo.pop_front() {
                for (nx, ny) in neighbours(&grid, tile).into_iter().flatten() {
                    grid[index(nx, ny)] = dist + 1;
                    todo.push_back(((nx, ny), dist + 1));
                }
            }
            return Wavefront { grid, offscreen, jumps: Vec::new() };
        }
        // Jumps cost more than a step, and a `path_costs` tile more than a plain one: Dijkstra, a jump relaxing its
        // start from its end (the search runs from the targets back toward Link, so the step relaxed from `tile` is
        // Link's step onto it and pays its path cost). `best` holds tentative distances; `grid` the settled ones.
        let mut best = vec![i32::MAX; CELLS];
        let mut todo: BinaryHeap<Reverse<(i32, Tile)>> = starts.into_iter().map(|t| Reverse((0, t))).collect();
        let mut by_end: Vec<&Jump> = jumps.iter().filter(|j| in_bounds(j.from.0, j.from.1)).collect();
        by_end.sort_unstable_by_key(|j| j.to);
        while let Some(Reverse((dist, tile))) = todo.pop() {
            if in_bounds(tile.0, tile.1) {
                let i = index(tile.0, tile.1);
                // Settled before (the starts hold 0 from the beginning; a later pop is a stale entry).
                if dist > 0 && grid[i] >= 0 {
                    continue;
                }
                grid[i] = dist;
            }
            let onto = room.path_cost(tile.0, tile.1);
            let next = neighbours(&grid, tile).into_iter().flatten().map(|t| (t, 1 + onto));
            let first = by_end.partition_point(|j| j.to < tile);
            let jumped = by_end[first..].iter().take_while(|j| j.to == tile).map(|j| (j.from, j.cost + onto));
            for ((nx, ny), cost) in next.chain(jumped) {
                let i = index(nx, ny);
                if grid[i] >= 0 || impassible.contains(&(nx, ny)) || dist + cost >= best[i] {
                    continue;
                }
                best[i] = dist + cost;
                todo.push(Reverse((dist + cost, (nx, ny))));
            }
        }
        Wavefront { grid, offscreen, jumps: jumps.to_vec() }
    }

    /// The jumps that start on `tile` (skips the search knows; none without clips).
    pub fn jumps_from(&self, tile: Tile) -> impl Iterator<Item = &Jump> {
        self.jumps.iter().filter(move |j| j.from == tile)
    }

    /// A wavefront from explicit distances (tests, synthetic objectives).
    pub fn from_distances(distances: impl IntoIterator<Item = (Tile, i32)>) -> Wavefront {
        let mut w = Wavefront {
            grid: vec![-1; CELLS],
            offscreen: Vec::new(),
            jumps: Vec::new(),
        };
        for ((x, y), d) in distances {
            if in_bounds(x, y) {
                w.grid[index(x, y)] = d;
            } else if d == 0 {
                w.offscreen.push((x, y));
            }
        }
        w
    }

    /// Distance of `tile`, None when unreachable (`Wavefront.get`).
    pub fn get(&self, tile: Tile) -> Option<i32> {
        if in_bounds(tile.0, tile.1) {
            let v = self.grid[index(tile.0, tile.1)];
            (v >= 0).then_some(v)
        } else {
            self.offscreen.contains(&tile).then_some(0)
        }
    }

    /// Distance of `tile`, 1000 when unreachable (`Wavefront[tile]`).
    pub fn distance(&self, tile: Tile) -> i32 {
        self.get(tile).unwrap_or(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOOR: u8 = 0x74;

    /// A floor-only 4:0x30 with water on `water` tiles (column, row).
    fn room(water: impl IntoIterator<Item = Tile>, ladder: bool) -> Room {
        let mut tiles = [FLOOR; CELLS];
        for (c, r) in water {
            tiles[index(c, r)] = WATER_TILE;
        }
        Room::new(MapLocation::new(4, 0x30, false), &tiles, ladder)
    }

    fn rows(rows: std::ops::Range<i32>) -> Vec<Tile> {
        (0..COLS).flat_map(|c| rows.clone().map(move |r| (c, r))).collect()
    }

    /// One square of water from wall to wall (0x30's row 2): crossed N with the ladder, a wall without it; on the
    /// ladder Link can't turn.
    #[test]
    fn the_ladder_crosses_a_water_row_one_square_tall() {
        let (with, without) = (room(rows(8..10), true), room(rows(8..10), false));
        assert!(with.can_move(15, 9, Direction::N) && with.can_move(15, 8, Direction::N));
        assert!(!without.can_move(15, 9, Direction::N));
        assert!(!with.can_move(15, 8, Direction::E) && !with.can_move(15, 8, Direction::W));
        let north = with.wavefront(&[Target::Tile((15, 2))], &[], &[], false);
        assert!(north.get((15, 14)).is_some());
        assert!(without.wavefront(&[Target::Tile((15, 2))], &[], &[], false).get((15, 14)).is_none());
        // Pixel moves: Link stops at Y = $85 without the ladder, goes on with it (level-4.md).
        assert!(!without.can_link_move_from(0x78, 0x85, Direction::N));
        assert!(with.can_link_move_from(0x78, 0x85, Direction::N) && with.can_link_move_from(0x78, 0x7D, Direction::N));
        // On the ladder (feet over the water at Y = $7D) Link can't turn.
        assert!(!with.can_link_move_from(0x78, 0x7D, Direction::E) && !with.can_link_move_from(0x78, 0x7D, Direction::W));
    }

    /// Two squares of water: the ladder can't cross it (0x21's island).
    #[test]
    fn the_ladder_does_not_cross_two_squares() {
        let wide = room(rows(8..12), true);
        assert!((8..12).all(|r| !wide.can_move(15, r, Direction::N)));
        assert!(wide.wavefront(&[Target::Tile((15, 2))], &[], &[], false).get((15, 16)).is_none());
    }

    /// A single water square in the floor (0x31's door line): crossed E and W with the ladder, not without.
    #[test]
    fn the_ladder_crosses_a_single_square_sideways() {
        let square = [(10, 9), (11, 9), (10, 10), (11, 10)];
        let (with, without) = (room(square, true), room(square, false));
        assert!(with.can_move(8, 8, Direction::E) && with.can_move(12, 8, Direction::W));
        assert!(!without.can_move(8, 8, Direction::E) && !without.can_move(12, 8, Direction::W));
        // Overworld rooms and cellars keep water a wall.
        let mut tiles = [FLOOR; CELLS];
        square.iter().for_each(|&(c, r)| tiles[index(c, r)] = WATER_TILE);
        assert!(!Room::new(MapLocation::new(4, 0x60, true), &tiles, true).can_move(8, 8, Direction::E));
    }

    /// `path_costs`: a step onto a costed tile counts 1 + cost.  Straight N from (15, 16) to (15, 2) is 14 steps; a
    /// band of cost 1 on columns 15-16, rows 6-15 makes the way one column W (2 steps longer) cheaper, and every tile
    /// keeps a distance (Link standing in the band included).
    #[test]
    fn path_costs_turn_the_way_round_a_band() {
        let open = room([], false);
        let band = [crate::game_map::PathCost { cols: (15, 16), rows: (6, 15), cost: 1 }];
        let costed = room([], false).with_path_costs(&band);
        let north = [Target::Tile((15, 2))];
        let (plain, round) = (open.wavefront(&north, &[], &[], false), costed.wavefront(&north, &[], &[], false));
        assert_eq!(plain.get((15, 16)), Some(14));
        assert_eq!(round.get((15, 16)), Some(16), "W, up column 14, back E");
        assert_eq!(round.get((14, 15)), Some(14));
        assert_eq!(round.get((15, 10)), Some(10), "in the band: W onto column 14, then up");
        assert_eq!((costed.path_cost(15, 6), costed.path_cost(15, 5), costed.path_cost(14, 10)), (1, 0, 0));
        let twice = room([], false).with_path_costs(&[band[0], band[0]]);
        assert_eq!(twice.path_cost(16, 15), 2, "overlapping rectangles add up");
    }

    /// Round 26: 0x42's drained pond. The pond's water ($90/$95) blocks Link until the game raises its walkable limit
    /// to `$99` (asm:FluteSecret_CuePondPaletteRow); the room model follows the game's limit when it is above the
    /// overworld default and ignores a lower one (every start state holds `$89`) and the underworld's.
    #[test]
    fn a_raised_walkable_limit_opens_the_drained_pond() {
        let mut tiles = [0x24u8; CELLS];
        for r in 8..=13 {
            tiles[index(12, r)] = 0x90;
            tiles[index(13, r)] = 0x95;
        }
        let pond = Room::new(MapLocation::new(0, 0x42, false), &tiles, false);
        assert!(!pond.is_tile_walkable(12, 10) && !pond.can_move(12, 13, Direction::N));
        let drained = pond.clone().with_first_unwalkable(0x99);
        assert!(drained.is_tile_walkable(12, 10) && drained.can_move(12, 13, Direction::N));
        assert_eq!(drained.threshold(), 0x99);
        assert_eq!(pond.clone().with_first_unwalkable(0x89).threshold(), OW_FIRST_UNWALKABLE);
        assert_eq!(pond.with_first_unwalkable(0).threshold(), OW_FIRST_UNWALKABLE);
        let floor = room([], false).with_first_unwalkable(0x99);
        assert_eq!(floor.threshold(), UW_FIRST_UNWALKABLE, "the underworld keeps its own");
    }

    /// Round 26 (7:0x69, also 4:0x21 and 4:0x11): an opened N bomb hole's rim is an unwalkable tile ($8C/$8D at the
    /// door position), yet the game lets Link through by the door's type. Given as open, the hole is an exit the path
    /// search reaches; without it the N side is a wall; a door already walkable is not touched.
    #[test]
    fn an_opened_bomb_hole_is_a_doorway() {
        let mut tiles = [FLOOR; CELLS];
        for c in 0..COLS {
            for r in 0..4 {
                tiles[index(c, r)] = 0xF6;
            }
        }
        tiles[index(15, 2)] = 0x8C;
        tiles[index(16, 2)] = 0x8D;
        tiles[index(15, 3)] = 0x24;
        tiles[index(16, 3)] = 0x24;
        let shut = Room::new(MapLocation::new(7, 0x69, false), &tiles, false);
        assert!(!shut.exits.contains(Direction::N));
        assert!(shut.wavefront(&[Target::Dir(Direction::N)], &[], &[], false).get((15, 10)).is_none());
        let open = shut.with_open_doorways(&[Direction::N]);
        assert_eq!(open.exits.get(Direction::N), Some(&[(15, 0)][..]));
        assert!(open.wavefront(&[Target::Dir(Direction::N)], &[], &[], false).get((15, 10)).is_some());
        let floor = room([], false);
        let before = floor.exits.get(Direction::N).map(<[Tile]>::to_vec);
        assert_eq!(floor.with_open_doorways(&[Direction::N]).exits.get(Direction::N).map(<[Tile]>::to_vec), before);
    }
}
