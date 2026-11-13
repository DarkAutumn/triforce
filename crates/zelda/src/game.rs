//! The decoded game state at one frame.
//!
//! A [`GameView`] owns a copy of the 10,240-byte RAM image and decodes Link, enemies, items, projectiles and hazards
//! once ([`ObjectClass`] sorts the object slots).
//! The room model and bomb walls are decoded on first use.

use crate::enums::{
    AnimationKind, AnimationState, ArrowKind, BoomerangKind, Direction, MapLocation, ObjectClass,
    Position, SwordKind, Tile, enemy,
};
use crate::game_map::GameMap;
use crate::room::{Room, Tiles};
use crate::tables::{self, RAM_SIZE, SRAM_OFFSET};
use crate::vars::Var;
use std::sync::{Arc, OnceLock};

pub const MODE_GAME_OVER: u8 = 8;
pub const MODE_CELLAR: u8 = 9;
pub const MODE_CAVE: u8 = 11;
pub const MODE_DYING: u8 = 17;

const OBJ_BLOCK_ID: u8 = 0x68;
const PUSH_BLOCK_SLOT: usize = 11;

const ANIMATION_BEAMS_ACTIVE: u8 = 16;
const ANIMATION_BEAMS_HIT: u8 = 17;
const ANIMATION_MAGIC_ROD_ACTIVE: u8 = 0x80;
const ANIMATION_FLAME_ACTIVE: u8 = 0x22;
const ANIMATION_BOMBS_ACTIVE: u8 = 18;
const ANIMATION_BOMBS_EXPLODED: [u8; 2] = [19, 20];
const ANIMATION_ARROW_ACTIVE: u8 = 10;
const ANIMATION_ARROW_END: u8 = 21;
const ANIMATION_BOOMERANG_MIN: u8 = 0x10;
const ANIMATION_BOOMERANG_MAX: u8 = 0x57;

// NES doorway coordinates (asm:Doorway_RequiredCoords) and screen/room bounds (asm:Link_ScreenEdgeBounds,
// asm:Room_ObjBoundsUW).
const DOORWAY_REQUIRED_X: i32 = 0x78;
const DOORWAY_REQUIRED_Y: i32 = 0x8D;
const CAVE_PERSON_LINE_Y: i32 = 0x8E;
const OW_EDGE_N: i32 = 0x3D;
const OW_EDGE_W: i32 = 0x00;
const OW_EDGE_E: i32 = 0xF0;
const UW_BOUND_L: i32 = 0x21;
const UW_BOUND_R: i32 = 0xD0;
const UW_BOUND_T: i32 = 0x5E;
const UW_BOUND_B: i32 = 0xBD;

/// asm:GleeokDeadNeckMask: bit n set once neck n's head is cut off.
pub const GLEEOK_DEAD_NECK_MASK: usize = 0x511;
/// Each neck's segment X / Y arrays (asm:Gleeok_NeckXAddrsLo/Hi, asm:Gleeok_NeckYAddrsLo/Hi): index 0 is the base
/// (slot 1), index 4 the head (slot 5) while the neck is worked on.
pub const GLEEOK_NECK_X: [usize; 4] = [0x438, 0x452, 0x46C, 0x395];
pub const GLEEOK_NECK_Y: [usize; 4] = [0x445, 0x45F, 0x479, 0x3BD];
const GLEEOK_HEAD_SEGMENT: usize = 4;
/// The attached heads share slot 5's HP byte and invincibility timer (asm:Gleeok_Update@LoadNeckBytes).
pub const GLEEOK_HEAD_SLOT: usize = 5;

/// The attached heads of a Gleeok in slot 1 (types $42-$45, `type - $41` necks; zelda-asm `enemies.md` `# Gleeok`):
/// (neck, head position) for every neck whose head is not cut off.  Empty without a Gleeok or while he dies.
pub fn gleeok_heads(ram: &[u8]) -> Vec<(usize, Position)> {
    let id = ram[tables::OBJ_ID.0 + 1];
    let dying = (16..=19).contains(&ram[tables::OBJ_SPAWN_STATE.0 + 1]);
    if !(enemy::GLEEOK..=enemy::GLEEOK4).contains(&id) || dying {
        return Vec::new();
    }
    let dead = ram[GLEEOK_DEAD_NECK_MASK];
    (0..(id - 0x41) as usize)
        .filter(|n| dead & (1 << n) == 0)
        .map(|n| {
            let (x, y) = (
                ram[GLEEOK_NECK_X[n] + GLEEOK_HEAD_SEGMENT],
                ram[GLEEOK_NECK_Y[n] + GLEEOK_HEAD_SEGMENT],
            );
            (n, Position::new(x as i32, y as i32))
        })
        .collect()
}

/// Tiles an object of `dims` at `position` overlaps with Link's top-left tile (`link_overlap_tiles`).
pub fn overlap_tiles(position: Position, dims: (i32, i32)) -> Vec<Tile> {
    let (tx, ty) = position.tile();
    let mut out = Vec::with_capacity(((dims.0 + 1) * (dims.1 + 1)) as usize);
    for x in -1..dims.0 {
        for y in -1..dims.1 {
            out.push((tx + x, ty + y));
        }
    }
    out
}

/// Tiles an object of `dims` at `position` occupies (`self_tiles`).
pub fn self_tiles(position: Position, dims: (i32, i32)) -> Vec<Tile> {
    let (tx, ty) = position.tile();
    let mut out = Vec::with_capacity((dims.0 * dims.1) as usize);
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            out.push((tx + x, ty + y));
        }
    }
    out
}

/// Distance from Link and the normalized direction to an object (float32, as the torch original).
fn distance_and_vector(position: Position, link: Position) -> (f32, (f32, f32)) {
    let (dx, dy) = ((position.x - link.x) as f32, (position.y - link.y) as f32);
    let d = (dx * dx + dy * dy).sqrt();
    if d.abs() < 1e-5 {
        (0.0, (0.0, 0.0))
    } else {
        (d, (dx / d, dy / d))
    }
}

/// Link (object slot 0).
#[derive(Clone, Debug, PartialEq)]
pub struct Link {
    pub position: Position,
    pub direction: Direction,
    pub status: u8,
}

impl Link {
    pub fn tile(&self) -> Tile {
        self.position.tile()
    }

    pub fn dims(&self) -> (i32, i32) {
        (2, 2)
    }
}

/// A monster in slots 1-11 ([`ObjectClass::Monster`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Enemy {
    pub index: u8,
    pub id: u8,
    pub position: Position,
    pub direction: Direction,
    pub health: i32,
    pub stun_timer: i32,
    pub spawn_state: u8,
    pub status: u8,
    pub distance: f32,
    pub vector: (f32, f32),
}

impl Enemy {
    pub fn dims(&self) -> (i32, i32) {
        if self.id == enemy::AQUAMENTUS {
            (3, 3)
        } else {
            (2, 2)
        }
    }

    pub fn tile(&self) -> Tile {
        self.position.tile()
    }

    /// Hit enough to die but not yet removed: death sparkle metastates $10-$13.
    pub fn is_dying(&self) -> bool {
        (16..=19).contains(&self.spawn_state)
    }

    /// Targetable (and able to touch Link): surfaced Leevers/Zoras, an on-screen Wallmaster, else spawned.
    pub fn is_active(&self) -> bool {
        let status = self.status;
        match self.id {
            enemy::RED_LEEVER | enemy::BLUE_LEEVER => status == 3,
            enemy::ZORA => (2..=4).contains(&status),
            enemy::WALLMASTER => status == 1,
            _ => self.spawn_state == 0,
        }
    }

    pub fn is_stunned(&self) -> bool {
        self.stun_timer > 0
    }

    pub fn link_overlap_tiles(&self) -> Vec<Tile> {
        overlap_tiles(self.position, self.dims())
    }

    pub fn self_tiles(&self) -> Vec<Tile> {
        self_tiles(self.position, self.dims())
    }
}

/// A dropped item (object type $60; `id` is its status byte), or the room treasure (index and id -1).
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub index: i8,
    pub id: i16,
    pub position: Position,
    /// Lifetime left in half-frames (the game decrements it every other frame; 255 for the treasure).
    pub timer: u8,
    pub distance: f32,
    pub vector: (f32, f32),
}

impl Item {
    pub fn dims(&self) -> (i32, i32) {
        (1, 1)
    }

    pub fn tile(&self) -> Tile {
        self.position.tile()
    }

    pub fn link_overlap_tiles(&self) -> Vec<Tile> {
        overlap_tiles(self.position, self.dims())
    }
}

/// A monster shot (object types $53-$5C, [`ObjectClass::Shot`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Projectile {
    pub index: u8,
    pub id: u8,
    pub position: Position,
    pub direction: Direction,
    pub distance: f32,
    pub vector: (f32, f32),
}

impl Projectile {
    pub fn dims(&self) -> (i32, i32) {
        (2, 2)
    }

    pub fn tile(&self) -> Tile {
        self.position.tile()
    }

    /// Shield-blockable (rock, boomerang, arrow).
    pub fn blockable(&self) -> bool {
        use crate::enums::projectile::*;
        matches!(self.id, OCTOROK_ROCK | BOOMERANG | ARROW)
    }
}

/// An object that hurts Link but can't be fought ([`ObjectClass::Hazard`]: boulders and their generator, traps,
/// bubbles). Not an enemy: no Fight target, no kill when its slot empties.
#[derive(Clone, Debug, PartialEq)]
pub struct Hazard {
    pub index: u8,
    pub id: u8,
    pub position: Position,
    pub direction: Direction,
    pub distance: f32,
    pub vector: (f32, f32),
}

impl Hazard {
    pub fn dims(&self) -> (i32, i32) {
        (2, 2)
    }

    pub fn tile(&self) -> Tile {
        self.position.tile()
    }
}

/// A synthetic entity for an intact bombable dungeon wall (game.yaml `bomb_walls`), at the NES hotspot.
#[derive(Clone, Debug, PartialEq)]
pub struct BombWall {
    pub wall_direction: Direction,
    pub position: Position,
    pub distance: f32,
    pub vector: (f32, f32),
}

impl BombWall {
    pub fn dims(&self) -> (i32, i32) {
        (2, 2)
    }

    /// asm:Bomb_WallHotspotsX / asm:Bomb_WallHotspotsY.
    pub fn hotspot(direction: Direction) -> Position {
        match direction {
            Direction::N => Position::new(0x78, 0x5D),
            Direction::S => Position::new(0x78, 0xBD),
            Direction::W => Position::new(0x20, 0x8D),
            _ => Position::new(0xD0, 0x8D),
        }
    }
}

/// The decoded state of one frame.
pub struct GameView {
    ram: Box<[u8; RAM_SIZE]>,
    /// Emulated frames when this state was decoded (the session's frame counter).
    pub frame: u64,
    /// Link just left a cave (he can't walk straight back in).
    pub just_exited_cave: bool,
    pub link: Link,
    /// Monsters ([`ObjectClass::Monster`]) by distance from Link (ties keep slot order).
    pub enemies: Vec<Enemy>,
    pub items: Vec<Item>,
    /// Monster shots by distance from Link.
    pub projectiles: Vec<Projectile>,
    /// Hazards by distance from Link.
    pub hazards: Vec<Hazard>,
    map: Arc<GameMap>,
    room: OnceLock<Room>,
    bomb_walls: OnceLock<Vec<BombWall>>,
}

impl std::fmt::Debug for GameView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GameView")
            .field("frame", &self.frame)
            .field("location", &self.full_location())
            .field("link", &self.link)
            .field(
                "enemies",
                &self.enemies.iter().map(|e| e.id).collect::<Vec<_>>(),
            )
            .finish()
    }
}

fn table(ram: &[u8; RAM_SIZE], t: (usize, usize), index: usize) -> u8 {
    ram[t.0 + index]
}

fn read_position(ram: &[u8; RAM_SIZE], index: usize) -> Position {
    Position::new(
        table(ram, tables::OBJ_POS_X, index) as i32,
        table(ram, tables::OBJ_POS_Y, index) as i32,
    )
}

fn read_direction(ram: &[u8; RAM_SIZE], index: usize) -> Direction {
    Direction::from_ram_value(table(ram, tables::OBJ_DIRECTION, index))
}

fn by_distance<T>(v: &mut [T], key: impl Fn(&T) -> f32) {
    v.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
}

impl GameView {
    /// Decodes the RAM image (CPU RAM followed by PRG-RAM).
    pub fn new(
        ram: &[u8; RAM_SIZE],
        frame: u64,
        just_exited_cave: bool,
        map: Arc<GameMap>,
    ) -> GameView {
        let ram = Box::new(*ram);
        let link = Link {
            position: read_position(&ram, 0),
            direction: read_direction(&ram, 0),
            status: table(&ram, tables::OBJ_STATUS, 0),
        };
        let (mut enemies, mut items, mut projectiles, mut hazards) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for index in 1..0xC {
            let id = table(&ram, tables::OBJ_ID, index);
            let position = read_position(&ram, index);
            let (distance, vector) = distance_and_vector(position, link.position);
            // Gleeok's neck base (slot 1) hurts and can't be killed: a hazard; his heads are added below.
            let class = match ObjectClass::of(id) {
                ObjectClass::Monster if index == 1 && (enemy::GLEEOK..=enemy::GLEEOK4).contains(&id) => {
                    ObjectClass::Hazard
                }
                class => class,
            };
            match class {
                ObjectClass::Item => items.push(Item {
                    index: index as i8,
                    id: table(&ram, tables::OBJ_STATUS, index) as i16,
                    position,
                    timer: table(&ram, tables::ITEM_TIMER, index),
                    distance,
                    vector,
                }),
                ObjectClass::Monster => enemies.push(Enemy {
                    index: index as u8,
                    id,
                    position,
                    direction: read_direction(&ram, index),
                    health: (table(&ram, tables::OBJ_HEALTH, index) >> 4) as i32,
                    stun_timer: table(&ram, tables::OBJ_STUN_TIMER, index) as i32,
                    spawn_state: table(&ram, tables::OBJ_SPAWN_STATE, index),
                    status: table(&ram, tables::OBJ_STATUS, index),
                    distance,
                    vector,
                }),
                ObjectClass::Shot => projectiles.push(Projectile {
                    index: index as u8,
                    id,
                    position,
                    direction: read_direction(&ram, index),
                    distance,
                    vector,
                }),
                ObjectClass::Hazard => hazards.push(Hazard {
                    index: index as u8,
                    id,
                    position,
                    direction: read_direction(&ram, index),
                    distance,
                    vector,
                }),
                _ => {}
            }
        }
        // Gleeok's attached heads aren't objects (slots 2-6 are type 0): one enemy per living neck, in slot 2 + neck,
        // all sharing slot 5's HP.
        let head_health = (table(&ram, tables::OBJ_HEALTH, GLEEOK_HEAD_SLOT) >> 4) as i32;
        for (neck, position) in gleeok_heads(&ram[..]) {
            let (distance, vector) = distance_and_vector(position, link.position);
            enemies.push(Enemy {
                index: 2 + neck as u8,
                id: enemy::GLEEOK_NECK_HEAD,
                position,
                direction: Direction::None,
                health: head_health,
                stun_timer: 0,
                spawn_state: 0,
                status: 0,
                distance,
                vector,
            });
        }
        by_distance(&mut enemies, |e| e.distance);
        by_distance(&mut items, |e| e.distance);
        by_distance(&mut projectiles, |e| e.distance);
        by_distance(&mut hazards, |e| e.distance);
        GameView {
            ram,
            frame,
            just_exited_cave,
            link,
            enemies,
            items,
            projectiles,
            hazards,
            map,
            room: OnceLock::new(),
            bomb_walls: OnceLock::new(),
        }
    }

    /// The RAM image (index `A` for CPU RAM, `A - 0x5800` for PRG-RAM).
    pub fn ram(&self) -> &[u8; RAM_SIZE] {
        &self.ram
    }

    /// A byte by CPU address ($0000-$07FF, $6000-$7FFF).
    pub fn peek(&self, addr: u16) -> u8 {
        match addr {
            0x0000..0x0800 => self.ram[addr as usize],
            0x6000..0x8000 => self.ram[addr as usize - SRAM_OFFSET],
            _ => panic!("address ${addr:04X} is outside RAM/PRG-RAM"),
        }
    }

    /// A data.json variable.
    pub fn var(&self, var: Var) -> i32 {
        var.decode(self.ram[var.addr() as usize])
    }

    fn byte(&self, var: Var) -> u8 {
        self.ram[var.addr() as usize]
    }

    pub fn game_map(&self) -> &Arc<GameMap> {
        &self.map
    }

    // --- location ---
    pub fn level(&self) -> u8 {
        self.byte(Var::Level)
    }

    pub fn location(&self) -> u8 {
        self.byte(Var::Location)
    }

    pub fn mode(&self) -> u8 {
        self.byte(Var::Mode)
    }

    pub fn in_cave(&self) -> bool {
        self.mode() == MODE_CAVE
    }

    /// In an underworld item cellar (mode 9, side view: walkways and ladders; up the left ladder leads out).
    pub fn in_cellar(&self) -> bool {
        self.mode() == MODE_CELLAR
    }

    /// The room; `in_cave` marks a sub-room, an overworld cave or an underworld cellar.
    pub fn full_location(&self) -> MapLocation {
        MapLocation::new(
            self.level(),
            self.location(),
            self.in_cave() || self.in_cellar(),
        )
    }

    pub fn game_over(&self) -> bool {
        matches!(self.mode(), MODE_DYING | MODE_GAME_OVER)
    }

    /// The room tiles (column-major `[32][22]`).
    pub fn current_tiles(&self) -> &Tiles {
        let (offset, len) = tables::TILE_LAYOUT;
        (&self.ram[offset..offset + len]).try_into().unwrap()
    }

    /// The room model built from the current tiles (with the ladder's crossings while Link holds it).
    pub fn room(&self) -> &Room {
        self.room
            .get_or_init(|| Room::new(self.full_location(), self.current_tiles(), self.ladder()))
    }

    // --- objects ---
    pub fn enemy_by_index(&self, index: u8) -> Option<&Enemy> {
        self.enemies.iter().find(|e| e.index == index)
    }

    pub fn item_by_index(&self, index: i8) -> Option<&Item> {
        self.items.iter().find(|e| e.index == index)
    }

    /// Enemies alive and active.
    pub fn active_enemies(&self) -> impl Iterator<Item = &Enemy> {
        self.enemies
            .iter()
            .filter(|e| e.is_active() && !e.is_dying())
    }

    pub fn has_active_enemies(&self) -> bool {
        self.active_enemies().next().is_some()
    }

    /// The room treasure position while it is on screen (treasure_flag == 0).
    pub fn treasure_location(&self) -> Option<Position> {
        (self.byte(Var::TreasureFlag) == 0)
            .then(|| Position::new(self.var(Var::TreasureX), self.var(Var::TreasureY)))
    }

    /// The room treasure as an item (index and id -1, timer 255).
    pub fn treasure(&self) -> Option<Item> {
        self.treasure_location().map(|position| {
            let (distance, vector) = distance_and_vector(position, self.link.position);
            Item {
                index: -1,
                id: -1,
                position,
                timer: 255,
                distance,
                vector,
            }
        })
    }

    /// An overworld room's game.yaml `bomb_secret` rock wall is still closed (no cave mouth in the room's tiles):
    /// a bomb target, as an intact dungeon bomb wall is.
    pub fn ow_bomb_secret_closed(&self) -> bool {
        self.level() == 0
            && !self.in_cave()
            && self.room().cave_tile.is_none()
            && self.map.get(self.full_location()).is_some_and(|r| r.bomb_secret.is_some())
    }

    /// A bomb-wall door ($4) the game has opened (room flag or CurOpenedDoors bit): its hole is passable whatever
    /// tile sits at the door position (the N and W holes' rims are unwalkable tiles; asm:Doorway_CheckLink goes by
    /// type).
    pub fn is_bomb_door_open(&self, direction: Direction) -> bool {
        self.level() != 0
            && crate::room_state::decode_room_state(&self.ram)
                .door(direction)
                .is_some_and(|d| d.door_type == crate::room_state::DoorType::Bomb && d.is_open)
    }

    /// Intact bombable walls of the current dungeon room (game.yaml `bomb_walls`).
    pub fn bomb_walls(&self) -> &[BombWall] {
        self.bomb_walls.get_or_init(|| {
            if self.level() == 0 {
                return Vec::new();
            }
            let Some(room) = self.map.get(self.full_location()) else {
                return Vec::new();
            };
            let tiles = self.current_tiles();
            room.bomb_walls
                .iter()
                .filter_map(|s| Direction::from_exit(s))
                .filter(|&d| self.room().is_wall_intact(d, tiles) && !self.is_bomb_door_open(d))
                .map(|d| {
                    let position = BombWall::hotspot(d);
                    let (distance, vector) = distance_and_vector(position, self.link.position);
                    BombWall {
                        wall_direction: d,
                        position,
                        distance,
                        vector,
                    }
                })
                .collect()
        })
    }

    /// A push block that hasn't been pushed yet (block object state 0, asm:PushBlock_Update_JumpTable).
    pub fn push_block_pending(&self) -> bool {
        table(&self.ram, tables::OBJ_ID, PUSH_BLOCK_SLOT) == OBJ_BLOCK_ID
            && table(&self.ram, tables::OBJ_STATUS, PUSH_BLOCK_SLOT) & 0x03 == 0
    }

    /// The 2x2 tiles of the pending push block (object Y is grid aligned: row = (y - $40) / 8).
    pub fn push_block_tiles(&self) -> Vec<Tile> {
        if !self.push_block_pending() {
            return Vec::new();
        }
        let col = table(&self.ram, tables::OBJ_POS_X, PUSH_BLOCK_SLOT) as i32 / 8;
        let row =
            (table(&self.ram, tables::OBJ_POS_Y, PUSH_BLOCK_SLOT) as i32 - 0x40).div_euclid(8);
        vec![
            (col, row),
            (col, row + 1),
            (col + 1, row),
            (col + 1, row + 1),
        ]
    }

    /// Whether a MOVE in `direction` pushes the pending push block: Link touches it, the room is clear
    /// (asm:IsRoomAllDead $34D) and he is aligned the way asm:PushBlock_State0Idle counts a push (same X for N/S,
    /// Y + 3 == block Y for E/W).
    pub fn is_push_block_direction(&self, direction: Direction) -> bool {
        let block = self.push_block_tiles();
        if block.is_empty() || self.peek(0x34D) == 0 {
            return false;
        }
        let (bx, by) = (
            table(&self.ram, tables::OBJ_POS_X, PUSH_BLOCK_SLOT) as i32,
            table(&self.ram, tables::OBJ_POS_Y, PUSH_BLOCK_SLOT) as i32,
        );
        let (x, y) = (self.link.position.x, self.link.position.y);
        let aligned = match direction {
            Direction::N | Direction::S => x == bx,
            _ => y + 3 == by,
        };
        aligned
            && Room::hotspot_tiles(x, y, direction)
                .iter()
                .any(|t| block.contains(t))
    }

    pub fn is_door_locked(&self, direction: Direction) -> bool {
        self.level() != 0 && Room::is_door_locked(direction, self.current_tiles())
    }

    pub fn is_door_barred(&self, direction: Direction) -> bool {
        self.level() != 0 && Room::is_door_barred(direction, self.current_tiles())
    }

    pub fn is_door_open(&self, direction: Direction) -> bool {
        !self.is_door_locked(direction) && !self.is_door_barred(direction)
    }

    /// Whether Link can move in `direction` (the NES flow of asm:Walker_Move).
    pub fn can_link_move(&self, direction: Direction) -> bool {
        let (px, py) = (self.link.position.x, self.link.position.y);
        if self.in_cave() && direction == Direction::N && py < CAVE_PERSON_LINE_Y {
            return false;
        }
        if self.var(Var::LinkGridOffset) != 0 {
            // Just out of a cave, a step short of the grid below the mouth, any move but S walks Link back in
            // (the game finishes the step along ObjDir first).
            if self.just_exited_cave {
                return direction == Direction::S;
            }
            return true;
        }
        let doorway_dir = self.byte(Var::DoorwayDir);
        if doorway_dir != 0
            && let Some(dw) = Direction::from_value(doorway_dir)
            && direction != dw
            && direction != dw.opposite()
        {
            return false;
        }
        // Overworld: on or one step before the N/W screen edge a move scrolls to the next room
        // (asm:Link_CheckScreenEdge).
        if self.level() == 0 && leaves_ow_screen(px, py, direction) {
            return true;
        }
        // A cellar has no doorways or room bounds (asm:Walker_Move skips both in mode 9): only its tiles block.
        let underworld_room = self.level() > 0 && !self.in_cellar();
        if underworld_room && direction == Direction::N && py <= OW_EDGE_N {
            return px == DOORWAY_REQUIRED_X && self.is_door_open(Direction::N);
        }
        if underworld_room && doorway_dir == 0 && at_dungeon_bound(px, py, direction) {
            return self.can_pass_doorway(px, py, direction);
        }
        let room = self.room();
        // Overworld past the right bound (where a screen wrap or an aborted one leaves Link, X $F8): no look-ahead
        // at X >= $F0, the tile under the hotspot decides (asm:Tile_GetAtObjOffset@AdjustX; mechanics.md "After the
        // wrap"). On the bound itself the edge test fires whatever the tile (asm:Link_CheckScreenEdge).
        if self.level() == 0 && direction == Direction::E && px > OW_EDGE_E {
            return room.is_tile_walkable(px.div_euclid(8), (py - 53).div_euclid(8));
        }
        // There, up or down reads column $1F and the 22 bytes after the tile map ($67F0 on) as column $20
        // (asm:Tile_GetAtObjOffset@FetchTile, mechanics.md "Outside the play area").
        if self.level() == 0 && matches!(direction, Direction::N | Direction::S) && px.div_euclid(8) == 0x1F {
            let row = if direction == Direction::N { (py - 61).div_euclid(8) } else { (py - 45).div_euclid(8) };
            if (0..crate::room::ROWS).contains(&row) {
                let past = self.peek(0x6530 + (0x20 * crate::room::ROWS + row) as u16);
                let walkable = |t: u8| t < room.threshold() || crate::room::OW_WALKABLE_OVERRIDES.contains(&t);
                return walkable(room.tile(0x1F, row).max(past));
            }
        }
        if room.can_link_move_from(px, py, direction) {
            return true;
        }
        if room.is_cave_entry_direction(px, py, direction) && !self.just_exited_cave {
            return true;
        }
        false
    }

    pub(crate) fn can_pass_doorway(&self, px: i32, py: i32, direction: Direction) -> bool {
        match direction {
            Direction::N | Direction::S if px != DOORWAY_REQUIRED_X => return false,
            Direction::E | Direction::W if py != DOORWAY_REQUIRED_Y => return false,
            _ => {}
        }
        if self.room().is_door_tile_walkable(direction) || self.is_bomb_door_open(direction) {
            return true;
        }
        self.is_door_locked(direction) && (self.keys() > 0 || self.magic_key())
    }

    // --- Link: health ---
    pub fn hearts_and_containers(&self) -> u8 {
        self.byte(Var::HeartsAndContainers)
    }

    pub fn partial_hearts(&self) -> u8 {
        self.byte(Var::PartialHearts)
    }

    /// Heart containers.
    pub fn max_health(&self) -> i32 {
        ((self.hearts_and_containers() >> 4) & 0x0F) as i32 + 1
    }

    /// Health in hearts (2.5 = two and a half).
    pub fn health(&self) -> f64 {
        let hc = self.hearts_and_containers();
        let mut filled = (hc & 0x0F) as i32;
        let containers = ((hc >> 4) & 0x0F) as i32 + 1;
        let p = self.partial_hearts();
        let mut partial = if p >= 0x80 {
            1.0
        } else if p > 0 {
            0.5
        } else {
            0.0
        };
        if filled > containers {
            filled = containers;
            partial = 0.0;
        }
        if filled == containers {
            partial = 0.0;
        }
        (filled as f64 + partial).min(containers as f64)
    }

    pub fn is_health_full(&self) -> bool {
        (self.health() - self.max_health() as f64).abs() < 1e-9
    }

    /// Exact NES beam condition (asm:SwordShot_Make): containers-1 == filled and partial >= $80.
    fn is_health_full_for_beams(&self) -> bool {
        let hc = self.hearts_and_containers();
        (hc >> 4) & 0x0F == hc & 0x0F && self.partial_hearts() >= 0x80
    }

    // --- Link: inventory ---
    pub fn sword(&self) -> SwordKind {
        SwordKind::from_value(self.byte(Var::Sword))
    }

    pub fn arrows(&self) -> ArrowKind {
        match self.byte(Var::Arrows) {
            0 => ArrowKind::None,
            1 => ArrowKind::Wood,
            _ => ArrowKind::Silver,
        }
    }

    pub fn boomerang(&self) -> BoomerangKind {
        if self.byte(Var::MagicBoomerang) != 0 {
            BoomerangKind::Magic
        } else if self.byte(Var::RegularBoomerang) != 0 {
            BoomerangKind::Wood
        } else {
            BoomerangKind::None
        }
    }

    pub fn bow(&self) -> bool {
        self.byte(Var::Bow) != 0
    }

    pub fn raft(&self) -> bool {
        self.byte(Var::Raft) != 0
    }

    pub fn ladder(&self) -> bool {
        self.byte(Var::StepLadder) != 0
    }

    pub fn rupees(&self) -> i32 {
        self.var(Var::Rupees)
    }

    pub fn bombs(&self) -> i32 {
        self.var(Var::Bombs)
    }

    /// Damage-free kills toward the forced drop (`$50` HelpDropCount, 0-10; asm:Damage_KillMonster).  Any hit to
    /// Link clears it; the next drop-capable death after the 10th takes the forced drop and clears it.
    pub fn help_drop_count(&self) -> i32 {
        self.var(Var::HelpDropCount)
    }

    /// The forced drop is bombs (`$51` HelpDropValue non-zero): the 10th kill was a bomb kill; else 5 rupees.
    pub fn help_drop_is_bombs(&self) -> bool {
        self.var(Var::HelpDropValue) != 0
    }

    /// The next kill is the 10th damage-free one: made with a bomb, it arms a forced bomb drop (4 bombs).
    pub fn next_kill_arms_drop(&self) -> bool {
        self.help_drop_count() == 9
    }

    /// Kills since Link was last hurt (`$627` WorldKillCount); a drop-capable death at exactly 16 drops a fairy,
    /// which also clears an armed forced drop.
    pub fn kill_streak(&self) -> i32 {
        self.var(Var::KillStreak)
    }

    pub fn keys(&self) -> i32 {
        self.var(Var::Keys)
    }

    pub fn magic_key(&self) -> bool {
        self.byte(Var::MagicKey) != 0
    }

    pub fn magic_shield(&self) -> bool {
        self.byte(Var::MagicShield) != 0
    }

    pub fn room_kills(&self) -> i32 {
        self.var(Var::RoomKills)
    }

    /// Triforce pieces (Triforce of Power not included).
    pub fn triforce_pieces(&self) -> u32 {
        self.byte(Var::Triforce).count_ones()
    }

    pub fn triforce_of_power(&self) -> bool {
        self.byte(Var::TriforceOfPower) != 0
    }

    /// Whether Link holds the triforce piece of `level` (1-8).
    pub fn has_triforce(&self, level: u8) -> bool {
        level >= 1 && self.byte(Var::Triforce) as u32 & crate::room_state::level_bit(level) != 0
    }

    /// The current dungeon's compass (level bit of $667, $669 in level 9).
    pub fn has_compass(&self) -> bool {
        match self.level() {
            0 => false,
            9 => self.byte(Var::Compass9) != 0,
            l => self.byte(Var::Compass) as u32 & crate::room_state::level_bit(l) != 0,
        }
    }

    /// The current dungeon's map (level bit of $668, $66A in level 9).
    pub fn has_map(&self) -> bool {
        match self.level() {
            0 => false,
            9 => self.byte(Var::Map9) != 0,
            l => self.byte(Var::Map) as u32 & crate::room_state::level_bit(l) != 0,
        }
    }

    // --- Link: weapons ---
    /// Action kinds Link's inventory allows, indexed by `ActionKind` (SWORD and BEAMS are exclusive; CLIP is the
    /// session's, `crate::clips`).
    pub fn available_actions(&self) -> [bool; crate::actions::NUM_KINDS] {
        use crate::enums::ActionKind as K;
        let mut a = [false; crate::actions::NUM_KINDS];
        a[K::Move.index()] = true;
        a[K::Wait.index()] = true;
        if self.sword() != SwordKind::None {
            a[if self.has_beams() { K::Beams } else { K::Sword }.index()] = true;
        }
        if self.bombs() != 0 {
            a[K::Bomb.index()] = true;
        }
        if self.arrows() != ArrowKind::None && self.rupees() > 0 && self.bow() {
            a[K::Arrow.index()] = true;
        }
        if self.boomerang() != BoomerangKind::None {
            a[K::Boomerang.index()] = true;
        }
        a
    }

    /// Link can fire sword beams.
    pub fn has_beams(&self) -> bool {
        if self.sword() == SwordKind::None
            || !self.is_health_full_for_beams()
            || self.is_sword_screen_locked()
        {
            return false;
        }
        self.animation_state(AnimationKind::Beams) == AnimationState::Inactive
    }

    pub fn are_beams_available(&self) -> bool {
        self.animation_state(AnimationKind::Beams) == AnimationState::Inactive && self.has_beams()
    }

    pub fn is_sword_screen_locked(&self) -> bool {
        self.sword_directions().is_empty()
    }

    /// Directions the sword (A) can strike in by buttons from here ([`crate::attack::plan`]): the press, or the
    /// turn before it, inside asm:Link_FilterInput's border bands; no sideways turn where the game forbids it.
    pub fn sword_directions(&self) -> Vec<Direction> {
        self.attack_directions(crate::enums::ActionKind::Sword)
    }

    /// Directions an attack of `kind` can be made in by buttons from here ([`crate::attack::plan`]; the
    /// boomerang's diagonals included). Weapon readiness and inventory are [`Self::weapon_ready`]'s and
    /// [`Self::available_actions`]'.
    pub fn attack_directions(&self, kind: crate::enums::ActionKind) -> Vec<Direction> {
        let dirs: &[Direction] = if crate::actions::is_diagonal(kind) {
            &crate::actions::DIRECTIONS
        } else {
            &crate::actions::CARDINALS
        };
        dirs.iter().copied().filter(|&d| crate::attack::plan(self, kind, d).is_some()).collect()
    }

    /// The sword is blocked (a Bubble's touch: SwordBlockedLongTimer $4C, or asm:IsSwordBlocked $52E); the game
    /// ignores the A button meanwhile (asm:Link_HandleInput).
    pub fn is_sword_blocked(&self) -> bool {
        self.byte(Var::SwordBlockedLongTimer) != 0 || self.byte(Var::SwordBlocked) != 0
    }

    /// The game would start `kind`'s weapon now (Link_HandleInput and the *_Wield routines): Link is in his normal
    /// state ($AC = 0), the sword is not blocked and its slot ($B9) is free for SWORD/BEAMS, the boomerang ($BB) and
    /// arrow ($BE) slots are free or returning (bit 7), and Bomb_Wield finds a bomb slot ($BC/$BD).
    pub fn weapon_ready(&self, kind: crate::enums::ActionKind) -> bool {
        use crate::enums::ActionKind as K;
        if self.link.status != 0 {
            return false;
        }
        let free_or_back = |s: u8| s == 0 || s & 0x80 != 0;
        match kind {
            K::Sword | K::Beams => !self.is_sword_blocked() && self.peek(0xB9) == 0,
            K::Boomerang => free_or_back(self.peek(0xBB)),
            K::Arrow => free_or_back(self.peek(0xBE)),
            K::Bomb => {
                let (s10, s11) = (self.peek(0xBC), self.peek(0xBD));
                let usable = |s: u8| s == 0 || s & 0xF0 != 0x10;
                let other = if usable(s10) {
                    s11
                } else if usable(s11) {
                    s10
                } else {
                    return false;
                };
                other == 0 || other >= 0x13
            }
            K::Move | K::Wait | K::Clip => true,
        }
    }

    /// State of a weapon animation slot.
    pub fn animation_state(&self, kind: AnimationKind) -> AnimationState {
        let active = |b: bool| {
            if b {
                AnimationState::Active
            } else {
                AnimationState::Inactive
            }
        };
        match kind {
            AnimationKind::Beams => match self.byte(Var::BeamAnimation) {
                ANIMATION_BEAMS_ACTIVE => AnimationState::Active,
                ANIMATION_BEAMS_HIT => AnimationState::Hit,
                _ => AnimationState::Inactive,
            },
            AnimationKind::Magic => active(
                self.byte(Var::BeamAnimation) == ANIMATION_MAGIC_ROD_ACTIVE
                    || self.byte(Var::BombOrFlameAnimation) == ANIMATION_FLAME_ACTIVE,
            ),
            AnimationKind::Flame1 => {
                active(self.byte(Var::BombOrFlameAnimation) == ANIMATION_FLAME_ACTIVE)
            }
            AnimationKind::Flame2 => {
                active(self.byte(Var::BombOrFlameAnimation2) == ANIMATION_FLAME_ACTIVE)
            }
            AnimationKind::Bomb1 => bomb_state(self.byte(Var::BombOrFlameAnimation)),
            AnimationKind::Bomb2 => bomb_state(self.byte(Var::BombOrFlameAnimation2)),
            AnimationKind::Arrow => active(
                (ANIMATION_ARROW_ACTIVE..=ANIMATION_ARROW_END)
                    .contains(&self.byte(Var::ArrowMagicAnimation)),
            ),
            AnimationKind::Boomerang => active(
                (ANIMATION_BOOMERANG_MIN..=ANIMATION_BOOMERANG_MAX)
                    .contains(&self.byte(Var::BaitOrBoomerangAnimation)),
            ),
            AnimationKind::Bait => panic!("animation state of {kind:?} is not implemented"),
        }
    }
}

fn bomb_state(value: u8) -> AnimationState {
    if value == ANIMATION_BOMBS_ACTIVE {
        AnimationState::Active
    } else if ANIMATION_BOMBS_EXPLODED.contains(&value) {
        AnimationState::Hit
    } else {
        AnimationState::Inactive
    }
}

/// On or one step before an overworld screen edge, moving on scrolls (asm:Link_ScreenEdgeBounds,
/// asm:Link_CheckScreenEdge, which runs even when the tile ahead is unwalkable).
fn leaves_ow_screen(px: i32, py: i32, direction: Direction) -> bool {
    match direction {
        Direction::N => py - 8 <= OW_EDGE_N,
        Direction::W => px > OW_EDGE_W && px - 8 <= OW_EDGE_W,
        _ => false,
    }
}

/// Dungeon room bounds (asm:Room_ObjBoundsUW).
pub(crate) fn at_dungeon_bound(px: i32, py: i32, direction: Direction) -> bool {
    match direction {
        Direction::N => py < UW_BOUND_T,
        Direction::S => py >= UW_BOUND_B,
        Direction::W => px < UW_BOUND_L,
        Direction::E => px >= UW_BOUND_R,
        _ => false,
    }
}
