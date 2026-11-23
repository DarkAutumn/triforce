//! Schema 8's `entity_threat` (`env::observation::threat_block`) on hand-built states: the RAM bytes the block reads
//! written directly, an open (or walled) tile grid, the entity block's presence and position. A monster walking at
//! Link, one across the room, a shot in flight, a wall in between, a shooter facing Link with and without its shot
//! out (round40.md "Observation v7's danger block").

use env::observation::{
    ENTITY_FEATURE_SIZE, ENTITY_THREAT_SIZE, NUM_ENTITY_SLOTS, ThreatParams, feature_names, threat_block,
};
use zelda::tiles::{Grid, TileClass};

const H: usize = 16;
const PARAMS: ThreatParams = ThreatParams { horizon: H, lane_slack: 0.0, invincibility: true };
/// Link's corner: his middle is (108, 128).
const LINK: (u8, u8) = (100, 120);

struct Room {
    ram: Vec<u8>,
    grid: Grid,
    feats: [[f32; ENTITY_FEATURE_SIZE]; NUM_ENTITY_SLOTS],
}

/// An object in RAM slot `slot` (observation slot `slot - 1`).
struct Obj {
    slot: usize,
    kind: u8,
    x: u8,
    y: u8,
    dir: u8,
    qspeed: u8,
    state: u8,
    shoot_timer: u8,
}

impl Obj {
    fn new(slot: usize, kind: u8, x: u8) -> Obj {
        Obj { slot, kind, x, y: LINK.1, dir: 0, qspeed: 0, state: 0, shoot_timer: 0 }
    }
    fn moving(mut self, dir: u8, qspeed: u8) -> Obj {
        (self.dir, self.qspeed) = (dir, qspeed);
        self
    }
}

impl Room {
    fn new() -> Room {
        let mut ram = vec![0u8; 0x800];
        (ram[0x70], ram[0x84]) = LINK;
        let grid = [[TileClass::Walkable as u8; 32]; 22];
        Room { ram, grid, feats: [[0.0; ENTITY_FEATURE_SIZE]; NUM_ENTITY_SLOTS] }
    }

    fn put(mut self, o: Obj) -> Room {
        let s = o.slot;
        self.ram[0x34F + s] = o.kind;
        self.ram[0x70 + s] = o.x;
        self.ram[0x84 + s] = o.y;
        self.ram[0x98 + s] = o.dir;
        self.ram[0x3BC + s] = o.qspeed;
        self.ram[0xAC + s] = o.state;
        self.ram[0x451 + s] = o.shoot_timer;
        let names = feature_names();
        let col = |n: &str| names.entity.iter().position(|x| x == n).unwrap();
        let f = &mut self.feats[s - 1];
        f[col("presence")] = 1.0;
        f[col("x")] = o.x as f32 / 255.0;
        f[col("y")] = (o.y as f32 - 64.0) / 176.0;
        self
    }

    /// A wall over tile column `c` (every row), or tile row `r` (every column).
    fn wall_column(mut self, c: usize) -> Room {
        self.grid.iter_mut().for_each(|row| row[c] = TileClass::Wall as u8);
        self
    }
    fn wall_row(mut self, r: usize) -> Room {
        self.grid[r] = [TileClass::Wall as u8; 32];
        self
    }

    /// Observation slot `i`'s columns, by name.
    fn threat(&self, i: usize) -> impl Fn(&str) -> f32 + use<> {
        let block = threat_block(&self.ram, &self.grid, &self.feats, &[false; NUM_ENTITY_SLOTS], PARAMS);
        let row: [f32; ENTITY_THREAT_SIZE] = block[i];
        let names = &feature_names().entity_threat;
        move |n: &str| row[names.iter().position(|x| x == n).unwrap()]
    }
}

fn soon(t: usize) -> f32 {
    ((H + 1 - t) as f64 / (H + 1) as f64) as f32
}

#[test]
fn a_monster_walking_at_link_touches_him_soon() {
    // Its middle 24 px right of his, walking left at 1 px a frame.
    let room = Room::new().put(Obj::new(1, 0x2A, 124).moving(2, 0x40));
    let t = room.threat(0);
    assert_eq!(t("soon_stand"), soon(16), "24 - t < 9 first at t = 16");
    assert_eq!(t("soon_E"), soon(7), "walking into it: 24 - 2.5 t < 9 at t = 7");
    assert_eq!(t("soon_W"), 0.0, "walking away at 1.5 outruns it");
    assert_eq!(t("near_W"), 15.0 / 16.0, "the closest gap is now's: 24 - 9");
    assert_eq!(t("near_stand"), 0.0, "touching by the horizon");
    assert_eq!((t("lane_ready"), t("lane_in"), t("lane_soon")), (0.0, 0.0, 0.0), "not a shooter");
}

#[test]
fn a_monster_across_the_room_is_not_soon() {
    let room = Room::new().put(Obj::new(1, 0x2A, 230).moving(2, 0x40));
    let t = room.threat(0);
    for m in ["stand", "N", "S", "W", "E"] {
        assert_eq!(t(&format!("soon_{m}")), 0.0, "{m}");
        assert_eq!(t(&format!("near_{m}")), 1.0, "{m}: farther than 16 px all the horizon");
    }
}

#[test]
fn a_shot_in_flight_is_seen_and_a_bouncing_one_is_not() {
    // A rock (3 px a frame) 30 px right of Link's middle, flying left.
    let mut rock = Obj::new(2, 0x53, 130).moving(2, 0xC0);
    rock.state = 0x10;
    let room = Room::new().put(rock);
    let t = room.threat(1);
    assert_eq!(t("soon_stand"), soon(8), "30 - 3 t < 9 first at t = 8");
    assert_eq!(t("soon_E"), soon(5), "walking into it: 30 - 4.5 t < 9 at t = 5");
    let mut parried = Obj::new(2, 0x53, 130).moving(2, 0xC0);
    parried.state = 0x30;
    let t = Room::new().put(parried).threat(1);
    assert_eq!((t("soon_stand"), t("near_stand")), (0.0, 1.0), "a bouncing shot harms nothing");
}

#[test]
fn a_wall_between_stops_the_walker_and_link() {
    // Link's corner at x 96 (middle 104); a walker at 124 (middle 132) walking left at 1.5 px a frame.
    let walker = || Obj::new(1, 0x2A, 124).moving(2, 0x60);
    let mut open = Room::new().put(walker());
    open.ram[0x70] = 96;
    let t = open.threat(0);
    assert_eq!(t("soon_stand"), soon(13), "28 - 1.5 t < 9 first at t = 13");
    assert_eq!(t("soon_E"), soon(7), "28 - 3 t < 9 at t = 7");
    // A wall over tile column 14 (x 112-119): the walker stops at x 120, Link cannot step east.
    let mut walled = Room::new().put(walker()).wall_column(14);
    walled.ram[0x70] = 96;
    let t = walled.threat(0);
    assert_eq!((t("soon_stand"), t("soon_E")), (0.0, 0.0), "nothing crosses the wall");
    assert_eq!(t("near_stand"), 15.0 / 16.0, "it stops 24 px off: gap 15");
}

#[test]
fn a_shooter_facing_link_threatens_his_lane_until_its_shot_is_out() {
    // A blue Octorok standing (its shoot timer running) facing left, Link ahead in its row.
    let octorok = |x: u8, timer: u8| {
        let mut o = Obj::new(1, 0x07, x).moving(2, 0);
        o.shoot_timer = timer;
        o
    };
    // 14 px ahead, the rock leaving next frame: it arrives at 1 + 1 + 6 / 3 = 4 frames; Link needs 6 to step out.
    let t = Room::new().put(octorok(114, 0x11)).threat(0);
    assert_eq!((t("lane_ready"), t("lane_in")), (1.0, 1.0));
    assert_eq!(t("lane_wait"), 1.0 / 32.0);
    assert_eq!(t("lane_soon"), soon(4));
    // 30 px ahead: 9.3 frames, he can still leave in 6, so not this decision.
    let t = Room::new().put(octorok(130, 0x11)).threat(0);
    assert_eq!((t("lane_in"), t("lane_soon")), (1.0, 0.0));
    // The same in a corridor (walls above and below his row): he cannot leave, the rock arrives in 10.
    let t = Room::new().put(octorok(130, 0x11)).wall_row(7).wall_row(9).threat(0);
    assert_eq!(t("lane_soon"), soon(10));
    // Across the room the rock is beyond the horizon.
    let t = Room::new().put(octorok(220, 0x11)).wall_row(7).wall_row(9).threat(0);
    assert_eq!((t("lane_in"), t("lane_soon")), (1.0, 0.0));
    // Its shot out (the timer past $10): no lane until it walks again.
    let t = Room::new().put(octorok(114, 0x08)).threat(0);
    assert_eq!((t("lane_ready"), t("lane_in"), t("lane_soon")), (0.0, 0.0, 0.0));
    // Facing away.
    let mut away = octorok(114, 0x11);
    away.dir = 1;
    let t = Room::new().put(away).threat(0);
    assert_eq!((t("lane_ready"), t("lane_in"), t("lane_soon")), (1.0, 0.0, 0.0));
    // A wall between stops a rock: Link is out of its reach.
    let t = Room::new().put(octorok(130, 0x11)).wall_row(7).wall_row(9).wall_column(15).threat(0);
    assert_eq!((t("lane_in"), t("lane_soon")), (0.0, 0.0));
    // A walking Octorok needs a 32-frame stop before it can shoot.
    let t = Room::new().put(octorok(114, 0)).threat(0);
    assert_eq!((t("lane_ready"), t("lane_wait"), t("lane_in"), t("lane_soon")), (1.0, 1.0, 1.0, 0.0));
}

#[test]
fn a_goriya_waiting_for_its_boomerang_has_no_lane() {
    // A blue Goriya 14 px ahead facing Link, its throw timer at 0 and wanting to throw: the boomerang (2.5 px a
    // frame) arrives at 0 + 1 + 6 / 2.5 = 3.4, frame 4.
    let goriya = |state: u8| Obj { state, ..Obj::new(1, 0x05, 114).moving(2, 0x20) };
    let wanting = |mut room: Room| {
        room.ram[0x412 + 1] = 1;
        room
    };
    let t = wanting(Room::new().put(goriya(0))).threat(0);
    assert_eq!((t("lane_ready"), t("lane_in"), t("lane_soon")), (1.0, 1.0, soon(4)));
    // Not wanting yet, 4 px past a grid point at 0.5 px a frame: the next grid point is 24 frames away.
    let mut room = Room::new().put(goriya(0));
    room.ram[0x394 + 1] = 4;
    let t = room.threat(0);
    assert_eq!((t("lane_wait"), t("lane_in"), t("lane_soon")), (24.0 / 32.0, 1.0, 0.0));
    // Waiting for it to come back (ObjState bit 7).
    let t = wanting(Room::new().put(goriya(0x80))).threat(0);
    assert_eq!((t("lane_ready"), t("lane_in"), t("lane_soon")), (0.0, 0.0, 0.0));
}

#[test]
fn no_touch_counts_while_link_is_invincible() {
    let walker = || Obj::new(1, 0x2A, 124).moving(2, 0x40);
    let mut room = Room::new().put(walker());
    room.ram[0x4F0] = 20;
    let t = room.threat(0);
    assert_eq!((t("soon_stand"), t("soon_E")), (0.0, 0.0), "the timer outlasts the horizon");
    let mut room = Room::new().put(walker());
    room.ram[0x4F0] = 10;
    let t = room.threat(0);
    assert_eq!(t("soon_stand"), soon(16), "touching at 16, after the timer");
    assert_eq!(t("soon_E"), soon(10), "overlapping from 7, harmful from 10");
}
