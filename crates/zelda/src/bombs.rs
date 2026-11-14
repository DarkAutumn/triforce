//! Bomb use per episode, room visit or chain room (Lee, Nov 11): bombs placed and what they did, the 10th
//! damage-free kills (each arms a forced drop: bombs when made with a bomb, else 5 rupees) and the bomb pickups.
//! And where a bomb opens a wall (round 22): [`BombTarget`], [`landing`].

use crate::change::Change;
use crate::enums::{ActionKind, Direction, Position};
use crate::game::{BombWall, GameView};
use std::ops::AddAssign;

/// Something a bomb opens, with the game's own reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BombTarget {
    /// An intact dungeon bomb wall: the bomb within $18 px of its hotspot on both axes when its blast ends
    /// (asm:Bomb_TryOpenWall; hotspots asm:Bomb_WallHotspotsX / asm:Bomb_WallHotspotsY, [`BombWall::hotspot`]).
    Wall(Direction),
    /// An overworld rock wall's tile object at this position (game.yaml `bomb_secret` tile (c, r): (8c, 8r + $40)):
    /// the blast's middle within $10 px of the object's on both axes (asm:RockWall_Update,
    /// asm:Collision_CheckTileObjWeapon; both middles +8, asm:Obj_GetMiddle), so the positions' difference.
    Rock(Position),
}

impl BombTarget {
    /// The rock wall of a game.yaml `bomb_secret` tile.
    pub fn rock(tile: crate::enums::Tile) -> BombTarget {
        BombTarget::Rock(Position::new(tile.0 * 8, tile.1 * 8 + 0x40))
    }

    /// A bomb lying at `bomb` opens this target.
    pub fn in_reach(&self, bomb: Position) -> bool {
        let near = |a: Position, reach: i32| (a.x - bomb.x).abs() < reach && (a.y - bomb.y).abs() < reach;
        match *self {
            BombTarget::Wall(d) => near(BombWall::hotspot(d), 0x18),
            BombTarget::Rock(at) => near(at, 0x10),
        }
    }
}

/// Where a BOMB in `d` lands from `state`: 16 px ahead of Link, facing `d`, on the frame its press comes after the
/// turn (asm:Link_PlaceWeaponAhead; [`crate::attack::press_link`]).  None where the bomb can't be placed that way.
pub fn landing(state: &GameView, d: Direction) -> Option<Position> {
    let link = crate::attack::press_link(state, ActionKind::Bomb, d)?;
    let (x, y) = (i32::from(link.x), i32::from(link.y));
    Some(match d {
        Direction::N => Position::new(x, y - 0x10),
        Direction::S => Position::new(x, y + 0x10),
        Direction::W => Position::new(x - 0x10, y),
        Direction::E => Position::new(x + 0x10, y),
        _ => return None,
    })
}

/// Counts over the steps of an episode, a visit or a chain room ([`BombStats::update`] per step).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BombStats {
    /// Bombs placed (BOMB actions that started a bomb).
    pub placed: u32,
    /// Enemy hits credited to the placing steps (sessions always look ahead for bombs: `crate::LookAhead`).
    pub hits: u32,
    /// Bombs placed that hit nothing.
    pub misses: u32,
    /// Damage the bombs dealt, in wood-sword hits (an enemy's HP >> 4): each credited hit's HP lost, a kill's
    /// remaining HP (Lee: how hard the bombs hit; a bomb takes 4 off, a Keese or Gel counts 1).
    pub damage: u32,
    /// 10th damage-free kills (`$50` 9 → 10), any weapon: the forced-drop opportunities taken.
    pub tenth_kills: u32,
    /// Bombs placed while the next kill was the 10th: tries for a forced bomb drop.
    pub tenth_tries: u32,
    /// Forced bomb drops armed (the 10th kill made with a bomb).
    pub armed: u32,
    /// Bomb pickups (Link's count went up within a step: a cheated bomb is granted outside the step).
    pub picked_up: u32,
}

impl BombStats {
    /// Counts one step.
    pub fn update(&mut self, change: &Change) {
        if change.placed_bomb() {
            let hits = change.hits() as u32;
            self.placed += 1;
            self.hits += hits;
            self.misses += u32::from(hits == 0);
            self.damage += change.enemies_hit.iter().map(|&(_, damage)| damage.max(0) as u32).sum::<u32>();
            self.tenth_tries += u32::from(change.previous.next_kill_arms_drop());
        }
        self.tenth_kills += u32::from(change.tenth_kill());
        self.armed += u32::from(change.armed_bomb_drop());
        self.picked_up += u32::from(
            change.previous.full_location() == change.state.full_location()
                && change.state.bombs() > change.previous.bombs(),
        );
    }

    /// `{placed, hits, misses, damage, tenth_kills, tenth_tries, armed, picked_up}`.
    pub fn fields(&self) -> [(&'static str, u32); 8] {
        [
            ("placed", self.placed),
            ("hits", self.hits),
            ("misses", self.misses),
            ("damage", self.damage),
            ("tenth_kills", self.tenth_kills),
            ("tenth_tries", self.tenth_tries),
            ("armed", self.armed),
            ("picked_up", self.picked_up),
        ]
    }
}

impl AddAssign for BombStats {
    fn add_assign(&mut self, o: BombStats) {
        self.placed += o.placed;
        self.hits += o.hits;
        self.misses += o.misses;
        self.damage += o.damage;
        self.tenth_kills += o.tenth_kills;
        self.tenth_tries += o.tenth_tries;
        self.armed += o.armed;
        self.picked_up += o.picked_up;
    }
}
