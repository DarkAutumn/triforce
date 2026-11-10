//! Room walkability, exits and the wavefront.
//!
//! Walkability uses the NES threshold system from GetCollidingTileMoving (Z_07.asm:2161-2320).

use crate::enums::{Direction, MapLocation, Tile};
use std::collections::{HashSet, VecDeque};

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

/// Tiles with value < threshold are walkable (Z_05.asm:6446-6450).
pub const OW_FIRST_UNWALKABLE: u8 = 0x89;
pub const UW_FIRST_UNWALKABLE: u8 = 0x78;

/// Overworld WalkableTiles substitutions (Z_07.asm:2137-2139).
pub const OW_WALKABLE_OVERRIDES: [u8; 9] = [0x8D, 0x91, 0x9C, 0xAC, 0xAD, 0xCC, 0xD2, 0xD5, 0xDF];

/// Cave entry tiles the agent may walk into (CheckPassiveTileObjects warps on the collision).
const OW_CAVE_ENTRY_TILES: [u8; 1] = [TOP_CAVE_TILE];

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

/// A room's walkability model.
#[derive(Clone, Debug)]
pub struct Room {
    pub location: MapLocation,
    tiles: Tiles,
    overworld: bool,
    threshold: u8,
    corridor: [bool; CELLS],
    loaded: bool,
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
    pub fn new(location: MapLocation, tiles: &Tiles) -> Room {
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
            exits: Exits::default(),
            cave_tile: None,
        };
        room.loaded = tiles
            .iter()
            .any(|&t| t < threshold || (overworld && ow_override(t)));
        if !overworld && !location.in_cave {
            room.compute_corridor_tiles();
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

    /// The top-left tiles of the room's stairs squares (tiles $70-$73; Z_05 CheckWarps sends Link down from a
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

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Raw tile value (column, row); panics out of bounds.
    pub fn tile(&self, tc: i32, tr: i32) -> u8 {
        self.tiles[index(tc, tr)]
    }

    pub fn tiles(&self) -> &Tiles {
        &self.tiles
    }

    /// NES walkability of a tile; off-screen is walkable (a room transition), as are doorway corridors.
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

    /// Whether Link at pixel (px, py) can move in `direction` (exact NES hotspots).
    pub fn can_link_move_from(&self, px: i32, py: i32, direction: Direction) -> bool {
        let feet_row = (py - 53).div_euclid(8);
        match direction {
            Direction::W => self.is_tile_walkable((px - 8).div_euclid(8), feet_row),
            Direction::E => self.is_tile_walkable((px + 16).div_euclid(8), feet_row),
            Direction::N => self.is_vertical_walkable(px.div_euclid(8), (py - 61).div_euclid(8)),
            Direction::S => self.is_vertical_walkable(px.div_euclid(8), (py - 45).div_euclid(8)),
            _ => false,
        }
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
        value < self.threshold
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
            Direction::E => tr + 1 < ROWS && self.is_tile_walkable(tc + 2, tr + 1),
            Direction::W => tr + 1 < ROWS && self.is_tile_walkable(tc - 1, tr + 1),
            Direction::S => tc + 1 < COLS && self.check_vertical(tc, tr + 2),
            Direction::N => tc + 1 < COLS && self.check_vertical(tc, tr),
            _ => false,
        }
    }

    fn check_vertical(&self, tc: i32, tr: i32) -> bool {
        self.is_tile_walkable(tc, tr) && self.is_tile_walkable(tc + 1, tr)
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
            exits.set(Direction::N, row(0), true);
            exits.set(Direction::S, row(ROWS - 2), true);
            exits.set(Direction::E, col(COLS - 1), true);
            exits.set(Direction::W, col(0), true);
        } else if self.location.in_cave {
            // An item cellar: up the ladder out of the top row (Z_05 CheckSubroom: Y < $40 moving up).
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

    /// The wavefront toward `targets` (`calculate_wavefront_for_link`).  `locked_doors` are treated as open.
    pub fn wavefront(
        &self,
        targets: &[Target],
        impassible: &[Tile],
        locked_doors: &[Direction],
    ) -> Wavefront {
        if locked_doors.is_empty() {
            return self.wavefront_inner(targets, impassible);
        }
        let mut room = self.clone();
        for &direction in locked_doors {
            room.add_locked_door_corridor(direction);
            room.exits.lists[exit_slot(direction)] = Some(vec![Self::door_exit_tile(direction)]);
        }
        room.wavefront_inner(targets, impassible)
    }

    fn wavefront_inner(&self, targets: &[Target], impassible: &[Tile]) -> Wavefront {
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
        Wavefront::compute(self, starts, &impassible)
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
}

impl Wavefront {
    fn compute(room: &Room, starts: HashSet<Tile>, impassible: &HashSet<Tile>) -> Wavefront {
        let mut grid = vec![-1i32; CELLS];
        let mut offscreen = Vec::new();
        let mut todo = VecDeque::new();
        let mut starts: Vec<Tile> = starts.into_iter().collect();
        starts.sort_unstable();
        for tile in starts {
            if in_bounds(tile.0, tile.1) {
                grid[index(tile.0, tile.1)] = 0;
            } else {
                offscreen.push(tile);
            }
            todo.push_back((tile, 0));
        }
        while let Some(((tx, ty), dist)) = todo.pop_front() {
            let source_offscreen = !in_bounds(tx, ty);
            for ((dx, dy), direction) in NEIGHBORS {
                let (nx, ny) = (tx + dx, ty + dy);
                if !in_bounds(nx, ny) || grid[index(nx, ny)] >= 0 || impassible.contains(&(nx, ny))
                {
                    continue;
                }
                if !source_offscreen && !room.can_move(nx, ny, direction) {
                    continue;
                }
                grid[index(nx, ny)] = dist + 1;
                todo.push_back(((nx, ny), dist + 1));
            }
        }
        Wavefront { grid, offscreen }
    }
    /// A wavefront from explicit distances (tests, synthetic objectives).
    pub fn from_distances(distances: impl IntoIterator<Item = (Tile, i32)>) -> Wavefront {
        let mut w = Wavefront {
            grid: vec![-1; CELLS],
            offscreen: Vec::new(),
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
