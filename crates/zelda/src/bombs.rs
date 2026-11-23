//! Bomb use per episode, room visit or chain room (the owner, Nov 11): bombs placed and what they did, the 10th
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

    /// A bomb of Link's already lies in reach of this target ([`live_bombs`]): it opens it when its blast ends (a
    /// wall, asm:Bomb_TryOpenWall) or during it (a rock wall, asm:RockWall_Update), so another bomb adds nothing
    /// until that one is gone (round 23: a second BOMB in the first one's fuse or blast at 4:0x21).
    pub fn awaits_bomb(&self, state: &GameView) -> bool {
        live_bombs(state).any(|b| self.in_reach(b))
    }
}

/// Where Link's bombs lie: object slots `$10` and `$11` in a bomb state, `$11` (placed) to `$14` (fading, after the
/// wall test); fires in the same slots (`$21`/`$22`) and empty slots are none (asm:Bomb_UpdateBombOrFire;
/// `ObjState` `$AC`, `ObjX` `$70`, `ObjY` `$84`).
pub fn live_bombs(state: &GameView) -> impl Iterator<Item = Position> + '_ {
    [0x10u16, 0x11].into_iter().filter(move |&slot| (0x11..=0x14).contains(&state.peek(0xAC + slot))).map(
        move |slot| Position::new(i32::from(state.peek(0x70 + slot)), i32::from(state.peek(0x84 + slot))),
    )
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
    /// remaining HP (the owner: how hard the bombs hit; a bomb takes 4 off, a Keese or Gel counts 1).
    pub damage: u32,
    /// 10th damage-free kills (`$50` 9 → 10), any weapon: the forced-drop opportunities taken.
    pub tenth_kills: u32,
    /// Bombs placed while the next kill was the 10th: tries for a forced bomb drop.
    pub tenth_tries: u32,
    /// Forced bomb drops armed (the 10th kill made with a bomb).
    pub armed: u32,
    /// Bomb pickups (Link's count went up within a step: a cheated bomb is granted outside the step).
    pub picked_up: u32,
    /// Dropped bombs (item `$00`) that appeared in the room (round 25: a boss's forced drop, the 10th kill's).
    pub dropped: u32,
    /// Dropped bombs taken: gone from the floor while their lifetime was not running out, or with Link's count up
    /// (taken at the bomb maximum counts too: the drop was spent).
    pub drop_taken: u32,
    /// Dropped bombs still on the floor when Link left the room (the exit step, or a maze loop's fresh copy).
    pub drop_left: u32,
    /// Dropped bombs whose lifetime ran out on the floor (asm:Drop_UpdateItem, about 510 frames).
    pub drop_expired: u32,
}

/// (slot, lifetime) of the dropped bombs lying in `state`'s room.
fn bomb_drops(state: &GameView) -> impl Iterator<Item = (i8, u8)> + '_ {
    state.items.iter().filter(|i| i.id == crate::enums::item::BOMBS).map(|i| (i.index, i.timer))
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
        self.update_drops(change);
    }

    /// The dropped bombs of one step (the way `crate::hearts::HeartTracker` follows hearts, from the step alone): a
    /// slot that holds a bomb drop it didn't hold (or holds a younger one) is a drop; one that is gone is taken when
    /// its lifetime could not have run out within the step or Link's count rose, else it expired; leaving the room
    /// leaves the ones on its floor.
    fn update_drops(&mut self, change: &Change) {
        let (prev, curr) = (&change.previous, &change.state);
        if change.entered_room() {
            self.drop_left += bomb_drops(prev).count() as u32;
            return;
        }
        let now = |slot: i8| bomb_drops(curr).find(|&(s, _)| s == slot).map(|(_, t)| t);
        let was = |slot: i8| bomb_drops(prev).find(|&(s, _)| s == slot).map(|(_, t)| t);
        let reach = change.frames / 2 + crate::hearts::TIMER_MARGIN;
        let gained = curr.bombs() > prev.bombs();
        for (slot, timer) in bomb_drops(prev) {
            if now(slot).is_none_or(|t| t > timer) {
                if u32::from(timer) > reach || gained {
                    self.drop_taken += 1;
                } else {
                    self.drop_expired += 1;
                }
            }
        }
        self.dropped += bomb_drops(curr).filter(|&(s, t)| was(s).is_none_or(|old| t > old)).count() as u32;
    }

    /// `{placed, hits, misses, damage, tenth_kills, tenth_tries, armed, picked_up, dropped, drop_taken, drop_left,
    /// drop_expired}`.
    pub fn fields(&self) -> [(&'static str, u32); 12] {
        [
            ("placed", self.placed),
            ("hits", self.hits),
            ("misses", self.misses),
            ("damage", self.damage),
            ("tenth_kills", self.tenth_kills),
            ("tenth_tries", self.tenth_tries),
            ("armed", self.armed),
            ("picked_up", self.picked_up),
            ("dropped", self.dropped),
            ("drop_taken", self.drop_taken),
            ("drop_left", self.drop_left),
            ("drop_expired", self.drop_expired),
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
        self.dropped += o.dropped;
        self.drop_taken += o.drop_taken;
        self.drop_left += o.drop_left;
        self.drop_expired += o.drop_expired;
    }
}
