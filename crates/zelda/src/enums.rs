//! Game enumerations and coordinates (`triforce/zelda_enums.py`).

use std::fmt;

/// Gameplay window top on screen: RAM y to tile row conversion (`GAMEPLAY_START_Y`).
pub const GAMEPLAY_START_Y: i32 = 56;

/// Action kinds of the v2 action schema (the value is the kind-head index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ActionKind {
    Move = 0,
    Wait = 1,
    Sword = 2,
    Beams = 3,
    Bomb = 4,
    Boomerang = 5,
    Arrow = 6,
}

impl ActionKind {
    pub const ALL: [ActionKind; 7] = [
        ActionKind::Move,
        ActionKind::Wait,
        ActionKind::Sword,
        ActionKind::Beams,
        ActionKind::Bomb,
        ActionKind::Boomerang,
        ActionKind::Arrow,
    ];

    pub fn from_index(index: usize) -> Option<ActionKind> {
        Self::ALL.get(index).copied()
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn name(self) -> &'static str {
        match self {
            ActionKind::Move => "MOVE",
            ActionKind::Wait => "WAIT",
            ActionKind::Sword => "SWORD",
            ActionKind::Beams => "BEAMS",
            ActionKind::Bomb => "BOMB",
            ActionKind::Boomerang => "BOOMERANG",
            ActionKind::Arrow => "ARROW",
        }
    }

    pub fn from_name(name: &str) -> Option<ActionKind> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// Kinds that use the B button with a selected item.
    pub const fn is_b_item(self) -> bool {
        matches!(
            self,
            ActionKind::Bomb | ActionKind::Boomerang | ActionKind::Arrow
        )
    }
}

/// Directions as the game encodes them (E1 W2 S4 N8, diagonals are unions).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Direction {
    None = 0,
    E = 1,
    W = 2,
    S = 4,
    N = 8,
    NE = 9,
    NW = 10,
    SE = 5,
    SW = 6,
}

impl Direction {
    pub const CARDINALS: [Direction; 4] = [Direction::N, Direction::S, Direction::W, Direction::E];

    pub const fn value(self) -> u8 {
        self as u8
    }

    /// Python `Direction(value)`: None for a value that is not a member.
    pub const fn from_value(value: u8) -> Option<Direction> {
        Some(match value {
            0 => Direction::None,
            1 => Direction::E,
            2 => Direction::W,
            4 => Direction::S,
            8 => Direction::N,
            9 => Direction::NE,
            10 => Direction::NW,
            5 => Direction::SE,
            6 => Direction::SW,
            _ => return None,
        })
    }

    /// `Direction.from_ram_value`: cardinals only, anything else is NONE.
    pub const fn from_ram_value(value: u8) -> Direction {
        match value {
            1 => Direction::E,
            2 => Direction::W,
            4 => Direction::S,
            8 => Direction::N,
            _ => Direction::None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Direction::None => "NONE",
            Direction::E => "E",
            Direction::W => "W",
            Direction::S => "S",
            Direction::N => "N",
            Direction::NE => "NE",
            Direction::NW => "NW",
            Direction::SE => "SE",
            Direction::SW => "SW",
        }
    }

    pub fn from_name(name: &str) -> Option<Direction> {
        [
            Direction::None,
            Direction::E,
            Direction::W,
            Direction::S,
            Direction::N,
            Direction::NE,
            Direction::NW,
            Direction::SE,
            Direction::SW,
        ]
        .into_iter()
        .find(|d| d.name() == name)
    }

    pub const fn is_cardinal(self) -> bool {
        matches!(
            self,
            Direction::N | Direction::S | Direction::E | Direction::W
        )
    }

    /// Opposite cardinal (NONE for anything else).
    pub const fn opposite(self) -> Direction {
        match self {
            Direction::N => Direction::S,
            Direction::S => Direction::N,
            Direction::E => Direction::W,
            Direction::W => Direction::E,
            _ => Direction::None,
        }
    }

    /// Unit vector (x right, y down); NONE and diagonals are (0, 0).
    pub const fn vector(self) -> (f32, f32) {
        match self {
            Direction::E => (1.0, 0.0),
            Direction::W => (-1.0, 0.0),
            Direction::S => (0.0, 1.0),
            Direction::N => (0.0, -1.0),
            _ => (0.0, 0.0),
        }
    }

    /// The direction for a game.yaml exit key ('N', 'S', 'E', 'W'); None for 'cave'.
    pub fn from_exit(name: &str) -> Option<Direction> {
        match name {
            "N" => Some(Direction::N),
            "S" => Some(Direction::S),
            "E" => Some(Direction::E),
            "W" => Some(Direction::W),
            _ => None,
        }
    }
}

/// Link's sword ($657).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SwordKind {
    None = 0,
    Wood = 1,
    White = 2,
    Magic = 3,
}

impl SwordKind {
    pub const fn from_value(value: u8) -> SwordKind {
        match value {
            0 => SwordKind::None,
            1 => SwordKind::Wood,
            2 => SwordKind::White,
            _ => SwordKind::Magic,
        }
    }
}

/// Link's arrows ($659).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArrowKind {
    None = 0,
    Wood = 1,
    Silver = 2,
}

/// Link's boomerang ($674 / $675).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BoomerangKind {
    None = 0,
    Wood = 1,
    Magic = 2,
}

/// Selected B item values ($656).
pub mod selected {
    pub const BOOMERANG: u8 = 0;
    pub const BOMBS: u8 = 1;
    pub const ARROWS: u8 = 2;
    pub const CANDLE: u8 = 4;
    pub const WHISTLE: u8 = 5;
    pub const FOOD: u8 = 6;
    pub const POTION: u8 = 7;
    pub const WAND: u8 = 8;
}

/// Sound bits of `sound_pulse_1` ($605).
pub mod sound {
    pub const ARROW_DEFLECTED: u8 = 0x01;
    pub const BOOMERANG_STUN: u8 = 0x02;
    pub const MAGIC_CAST: u8 = 0x04;
    pub const KEY_PICKUP: u8 = 0x08;
    pub const SMALL_HEART_PICKUP: u8 = 0x10;
    pub const SET_BOMB: u8 = 0x20;
    pub const HEART_WARNING: u8 = 0x40;
}

/// State of a weapon animation slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationState {
    Inactive,
    Active,
    Hit,
}

/// Weapon animation slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnimationKind {
    Beams,
    Bomb1,
    Bomb2,
    Flame1,
    Flame2,
    Magic,
    Arrow,
    Boomerang,
    Bait,
}

/// Enemy object types (`ZeldaEnemyKind`, UpdateObject_JumpTable Z_07.asm:5321).
pub mod enemy {
    pub const BLUE_LYNEL: u8 = 0x01;
    pub const RED_LYNEL: u8 = 0x02;
    pub const BLUE_MOBLIN: u8 = 0x03;
    pub const RED_MOBLIN: u8 = 0x04;
    pub const BLUE_GORIYA: u8 = 0x05;
    pub const RED_GORIYA: u8 = 0x06;
    pub const BLUE_OCTOROK: u8 = 0x07;
    pub const RED_OCTOROK: u8 = 0x08;
    pub const BLUE_SLOW_OCTOROK: u8 = 0x09;
    pub const RED_SLOW_OCTOROK: u8 = 0x0A;
    pub const BLUE_DARKNUT: u8 = 0x0B;
    pub const RED_DARKNUT: u8 = 0x0C;
    pub const BLUE_TEKTITE: u8 = 0x0D;
    pub const RED_TEKTITE: u8 = 0x0E;
    pub const BLUE_LEEVER: u8 = 0x0F;
    pub const RED_LEEVER: u8 = 0x10;
    pub const ZORA: u8 = 0x11;
    pub const VIRE: u8 = 0x12;
    pub const ZOL: u8 = 0x13;
    pub const GREEN_GEL: u8 = 0x14;
    pub const BLUE_GEL: u8 = 0x15;
    pub const POLS_VOICE: u8 = 0x16;
    pub const LIKE_LIKE: u8 = 0x17;
    pub const DIGDOGGER: u8 = 0x18;
    pub const DIGDOGGER_CHILD: u8 = 0x19;
    pub const PEAHAT: u8 = 0x1A;
    pub const BLUE_KEESE: u8 = 0x1B;
    pub const RED_KEESE: u8 = 0x1C;
    pub const BLACK_KEESE: u8 = 0x1D;
    pub const ARMOS: u8 = 0x1E;
    pub const GHINI: u8 = 0x21;
    pub const FLYING_GHINI: u8 = 0x22;
    pub const BLUE_WIZZROBE: u8 = 0x23;
    pub const RED_WIZZROBE: u8 = 0x24;
    pub const PATRA_CHILD1: u8 = 0x25;
    pub const PATRA_CHILD2: u8 = 0x26;
    pub const WALLMASTER: u8 = 0x27;
    pub const ROPE: u8 = 0x28;
    pub const STALFOS: u8 = 0x2A;
    pub const BUBBLE: u8 = 0x2B;
    pub const BLUE_BUBBLE: u8 = 0x2C;
    pub const RED_BUBBLE: u8 = 0x2D;
    pub const GIBDO: u8 = 0x30;
    pub const DODONGO: u8 = 0x31;
    pub const DODONGO2: u8 = 0x32;
    pub const BLUE_GOHMA: u8 = 0x33;
    pub const RED_GOHMA: u8 = 0x34;
    pub const DIGDOGGER2: u8 = 0x38;
    pub const DIGDOGGER3: u8 = 0x39;
    pub const BLUE_LAMNOLA: u8 = 0x3A;
    pub const RED_LAMNOLA: u8 = 0x3B;
    pub const MANHANDLA: u8 = 0x3C;
    pub const AQUAMENTUS: u8 = 0x3D;
    pub const GANON: u8 = 0x3E;
    pub const MOLDORM: u8 = 0x41;
    pub const GLEEOK: u8 = 0x42;
    pub const GLEEOK2: u8 = 0x43;
    pub const GLEEOK3: u8 = 0x44;
    pub const GLEEOK4: u8 = 0x45;
    pub const GLEEOK_HEAD: u8 = 0x46;
    pub const PATRA: u8 = 0x47;
    pub const PATRA2: u8 = 0x48;

    /// Boss object types (`critics.BOSS_KINDS`).
    pub const BOSS_KINDS: [u8; 16] = [
        AQUAMENTUS,
        DODONGO,
        DODONGO2,
        MANHANDLA,
        BLUE_GOHMA,
        RED_GOHMA,
        DIGDOGGER,
        DIGDOGGER2,
        DIGDOGGER3,
        GLEEOK,
        GLEEOK2,
        GLEEOK3,
        GLEEOK4,
        GLEEOK_HEAD,
        PATRA,
        GANON,
    ];

    pub fn is_boss(id: u8) -> bool {
        BOSS_KINDS.contains(&id)
    }
}

/// Dropped item ids (`ZeldaItemKind`; the room treasure has id -1).
pub mod item {
    pub const TREASURE: i16 = -1;
    pub const BOMBS: i16 = 0x00;
    pub const BLUE_RUPEE: i16 = 0x0F;
    pub const RUPEE: i16 = 0x18;
    pub const CLOCK: i16 = 0x21;
    pub const HEART: i16 = 0x22;
    pub const FAIRY: i16 = 0x23;
}

/// Hostile projectile ids (`ZeldaProjectileId`).
pub mod projectile {
    pub const OCTOROK_ROCK: u8 = 83;
    pub const ZORA_FIREBALL: u8 = 85;
    pub const ARROW: u8 = 91;
    pub const BOOMERANG: u8 = 92;
}

/// A tile (column, row) in the 32x22 play area; may lie outside it.
pub type Tile = (i32, i32);

/// A pixel position in game RAM coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub const fn new(x: i32, y: i32) -> Position {
        Position { x, y }
    }

    /// The tile of the position (floor division, like Python `//`).
    pub const fn tile(self) -> Tile {
        (
            self.x.div_euclid(8),
            (self.y - GAMEPLAY_START_Y).div_euclid(8),
        )
    }
}

/// A room: level, room id ($EB, row in the high nibble) and whether Link is in its cave.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MapLocation {
    pub level: u8,
    pub value: u8,
    pub in_cave: bool,
}

impl MapLocation {
    pub const fn new(level: u8, value: u8, in_cave: bool) -> MapLocation {
        MapLocation {
            level,
            value,
            in_cave,
        }
    }

    pub const fn x(self) -> i32 {
        (self.value & 0x0F) as i32
    }

    pub const fn y(self) -> i32 {
        ((self.value & 0xF0) >> 4) as i32
    }

    pub const fn from_coordinates(level: u8, x: i32, y: i32, in_cave: bool) -> MapLocation {
        MapLocation::new(level, ((y << 4) | x) as u8, in_cave)
    }

    /// The adjacent room in a cardinal direction (None off the 16x16 map or for other directions).
    pub fn location_in_direction(self, direction: Direction) -> Option<MapLocation> {
        let (mut x, mut y) = (self.x(), self.y());
        match direction {
            Direction::N => y -= 1,
            Direction::S => y += 1,
            Direction::E => x += 1,
            Direction::W => x -= 1,
            _ => return None,
        }
        ((0..=0xF).contains(&x) && (0..=0xF).contains(&y))
            .then(|| MapLocation::from_coordinates(self.level, x, y, self.in_cave))
    }

    /// Direction of movement from this room to `next` (`get_direction_to`).
    pub fn direction_to(self, next: MapLocation) -> Direction {
        if self.in_cave && !next.in_cave {
            return Direction::S;
        }
        if self.level == 0 && next.level != 0 {
            return Direction::S;
        }
        if self.level != 0 && next.level == 0 {
            return Direction::N;
        }
        if self.manhattan_distance(next) == 1 {
            if self.x() < next.x() {
                return Direction::E;
            } else if self.x() > next.x() {
                return Direction::W;
            } else if self.y() < next.y() {
                return Direction::S;
            } else if self.y() > next.y() {
                return Direction::N;
            }
        }
        Direction::None
    }

    /// Room distance (1 between a room and its cave, or across levels).
    pub fn manhattan_distance(self, other: MapLocation) -> i32 {
        if self.in_cave != other.in_cave || self.level != other.level {
            return 1;
        }
        (self.x() - other.x()).abs() + (self.y() - other.y()).abs()
    }
}

impl fmt::Display for MapLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, 0x{:02x}{})",
            self.level,
            self.value,
            if self.in_cave { ", cave" } else { "" }
        )
    }
}
