//! Attacks by buttons in the real game (round 21 movement cleanup; `zelda::attack`, `zelda::submenu`): the mask
//! offers exactly the strikes the controller can make, the executor makes them without writing memory, and the item
//! sub-screen selects every B item from every other one.
#![cfg(feature = "zcore")]

mod common;

use common::{load_named_state, open_core};
use parking_lot::Mutex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use zcore::{MachineState, ZCore};
use zelda::attack::{Refusal, facing_for};
use zelda::executor::{buttons, direction_buttons};
use zelda::{
    Action, ActionKind, Direction, GameMap, GameView, LoadOptions, LookAhead, Nes, Session, SessionSnapshot,
    compute_action_mask,
};

/// zcore counting every write to memory (`poke`): the executor must play attacks with the controller alone.
struct Poked(ZCore, Arc<AtomicU32>);

impl Nes for Poked {
    type State = MachineState;
    fn run_frame(&mut self, buttons: u8, render: bool) {
        self.0.run_frame(buttons, render);
    }
    fn ram(&self) -> &[u8; 0x800] {
        self.0.ram()
    }
    fn prg_ram(&self) -> &[u8; 0x2000] {
        self.0.prg_ram()
    }
    fn poke(&mut self, addr: u16, value: u8) {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.poke(addr, value);
    }
    fn snapshot(&self) -> MachineState {
        self.0.snapshot()
    }
    fn restore(&mut self, state: &MachineState) {
        self.0.restore(state);
    }
    fn poke_state(state: &mut MachineState, addr: u16, value: u8) {
        state.poke(addr, value);
    }
}

type S = Session<Poked>;
type Snap = SessionSnapshot<MachineState>;

const KINDS: [ActionKind; 4] = [ActionKind::Sword, ActionKind::Boomerang, ActionKind::Arrow, ActionKind::Bomb];

fn session() -> Option<(S, Arc<AtomicU32>)> {
    let pokes = Arc::new(AtomicU32::new(0));
    let mut s = Session::new(Poked(open_core()?, pokes.clone()), Arc::new(GameMap::builtin()));
    s.look_ahead = LookAhead::Off;
    Some((s, pokes))
}

/// Loads `name` with every B item (test setup, before any measurement), the clock (monsters stand still and Link
/// is invincible: no hit cuts a scan short) and the B item `selected`.
fn load(s: &mut S, name: &str) {
    s.load(&load_named_state(name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    let nes = s.nes_mut();
    for (addr, value) in [(0x658, 8), (0x659, 1), (0x65A, 1), (0x674, 1), (0x66D, 50), (0x66C, 1)] {
        nes.poke(addr, value);
    }
    s.apply_ram(&[]);
}

fn ram(s: &S) -> [u8; 0x800] {
    *s.nes().ram()
}

fn in_play(s: &S, room: (u8, u8, u8)) -> bool {
    let r = ram(s);
    (r[0x10], r[0xEB], r[0x12]) == room
}

fn room_of(s: &S) -> (u8, u8, u8) {
    let r = ram(s);
    (r[0x10], r[0xEB], r[0x12])
}

/// Positions Link takes from the start state: into the room, out through the entry door a frame at a time and back,
/// straight walks to every wall and a random walk (every frame a snapshot, in the same room and in play).
fn positions(s: &mut S, rng: &mut ChaCha8Rng, walk_frames: u32) -> Vec<Snap> {
    let room = room_of(s);
    let entry = s.snapshot();
    let mut spots = vec![entry.clone()];
    let push = |s: &mut S, spots: &mut Vec<Snap>| {
        if in_play(s, room) && ram(s)[0xAC] == 0 {
            spots.push(s.snapshot());
            true
        } else {
            false
        }
    };
    let pads = [buttons::UP, buttons::DOWN, buttons::LEFT, buttons::RIGHT];
    for &pad in &pads {
        s.restore(&entry);
        for _ in 0..90 {
            s.press(pad, 1);
            if !push(s, &mut spots) {
                break;
            }
        }
        // And back the other way from there, to the far side.
        let back = match pad {
            buttons::UP => buttons::DOWN,
            buttons::DOWN => buttons::UP,
            buttons::LEFT => buttons::RIGHT,
            _ => buttons::LEFT,
        };
        if let Some(last) = spots.last().cloned() {
            s.restore(&last);
            for _ in 0..30 {
                s.press(back, 1);
                if !push(s, &mut spots) {
                    break;
                }
            }
        }
    }
    s.restore(&entry);
    let mut good = entry.clone();
    let mut frames = 0;
    while frames < walk_frames {
        let pad = pads[rng.random_range(0..4)];
        let hold = rng.random_range(1..12);
        for _ in 0..hold {
            s.press(pad, 1);
            frames += 1;
            if push(s, &mut spots) {
                good = spots.last().unwrap().clone();
            } else {
                s.restore(&good);
                break;
            }
        }
    }
    let mut seen = BTreeSet::new();
    spots.retain(|snap| {
        s.restore(snap);
        let r = ram(s);
        seen.insert((r[0x70], r[0x84], r[0x98], r[0x394], r[0x53], r[0x3A8]))
    });
    spots
}

/// Whether the frames seen show a strike of `kind` in `d`: the sword in its hitting state facing `d` on every such
/// frame, or a new boomerang, arrow or bomb facing `d` on its first frame (an arrow fired into a wall is a spark at
/// once, a boomerang thrown into one turns back at once: both were thrown).
fn strike_seen(kind: ActionKind, d: Direction, frames: &[[u8; 0x800]]) -> bool {
    let want = d.value();
    let thrown = |slot: usize, dir: usize| {
        frames.windows(2).any(|w| w[0][slot] & 0x7F == 0 && w[1][slot] != 0 && w[1][slot] & 0x80 == 0 && w[1][dir] == want)
    };
    match kind {
        ActionKind::Sword | ActionKind::Beams => {
            let hitting: Vec<u8> = frames.iter().filter(|r| r[0xB9] & 0x0F == 2).map(|r| r[0xA5]).collect();
            !hitting.is_empty() && hitting.iter().all(|&dir| dir == want)
        }
        ActionKind::Boomerang => thrown(0xBB, 0xA7),
        ActionKind::Arrow => thrown(0xBE, 0xAA),
        ActionKind::Bomb => frames.windows(2).any(|w| {
            (0..2).any(|i| w[0][0xBC + i] & 0xF0 != 0x10 && w[1][0xBC + i] & 0xF0 == 0x10 && w[1][0xA8 + i] == want)
        }),
        _ => false,
    }
}

fn select(s: &mut S, kind: ActionKind) {
    if let Some(slot) = zelda::submenu::slot(kind) {
        s.nes_mut().poke(0x656, slot); // test setup: the selection itself is tested on its own
        s.apply_ram(&[]);
    }
}

/// What one attack did, watched frame by frame.
struct Played {
    struck: bool,
    written: u32,
    stayed: bool,
    change: zelda::Change,
}

/// Executes `kind d` from `spot` (the B item selected first by a test-setup write when `preselect`) and watches every
/// frame.
fn play(s: &mut S, pokes: &AtomicU32, spot: &Snap, kind: ActionKind, d: Direction, preselect: bool) -> Played {
    s.restore(spot);
    let room = room_of(s);
    if preselect {
        select(s, kind);
    }
    let seen = Arc::new(Mutex::new(vec![ram(s)]));
    let sink = seen.clone();
    s.set_frame_sink(Some(Box::new(move |n: &Poked| sink.lock().push(*n.ram()))));
    pokes.store(0, Ordering::Relaxed);
    let change = s.act(Action::new(kind, d, 0).unwrap()).unwrap();
    let written = pokes.load(Ordering::Relaxed);
    s.set_frame_sink(None);
    let struck = strike_seen(kind, d, &seen.lock());
    Played { struck, written, stayed: in_play(s, room), change }
}

/// Executes `kind d` from `spot` with its B item selected: (struck, memory written, same room and in play).
fn execute(s: &mut S, pokes: &AtomicU32, spot: &Snap, kind: ActionKind, d: Direction) -> (bool, u32, bool) {
    let p = play(s, pokes, spot, kind, d, true);
    (p.struck, p.written, p.stayed)
}

/// The fewest frames of holding the facing the strike needs after which the button (with the direction, or alone)
/// makes the strike from `spot` with Link still in the room, if any up to 32.
fn buttons_can(s: &mut S, spot: &Snap, kind: ActionKind, d: Direction) -> Option<u32> {
    let button = if kind == ActionKind::Sword { buttons::A } else { buttons::B };
    let turn = direction_buttons(facing_for(d));
    s.restore(spot);
    let room = room_of(s);
    select(s, kind);
    let base = s.snapshot();
    for k in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32] {
        for with_dir in [true, false] {
            s.restore(&base);
            let mut frames = vec![ram(s)];
            for _ in 0..k {
                s.press(turn, 1);
                frames.push(ram(s));
            }
            if !in_play(s, room) {
                continue;
            }
            let press = button | if with_dir { direction_buttons(d) } else { 0 };
            for b in [press, press, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] {
                s.press(b, 1);
                frames.push(ram(s));
            }
            if in_play(s, room) && strike_seen(kind, d, &frames) {
                return Some(k);
            }
        }
    }
    None
}

fn directions(kind: ActionKind) -> Vec<Direction> {
    let mut v = vec![Direction::N, Direction::S, Direction::W, Direction::E];
    if kind == ActionKind::Boomerang {
        v.extend([Direction::NW, Direction::NE, Direction::SW, Direction::SE]);
    }
    v
}

fn describe(s: &S) -> String {
    let r = ram(s);
    format!(
        "({:02X},{:02X}) facing {} offset {} frac {:02X} $53={}",
        r[0x70], r[0x84], r[0x98], r[0x394] as i8, r[0x3A8], r[0x53]
    )
}

/// Runs the scan over `states`; returns the problems (offered but not struck, written memory, left the room) and
/// the refused strikes plain buttons can make, by state.
fn scan(states: &[&str], per_state: usize, walk_frames: u32) -> (Vec<String>, BTreeMap<String, Vec<String>>, usize) {
    let Some((mut s, pokes)) = session() else { return (vec![], BTreeMap::new(), 0) };
    let mut rng = ChaCha8Rng::seed_from_u64(21);
    let mut problems = Vec::new();
    let mut workable = BTreeMap::new();
    let mut checked = 0;
    for &name in states {
        load(&mut s, name);
        let mut spots = positions(&mut s, &mut rng, walk_frames);
        // An even sample of the positions at or beside a dungeon doorway (in it, or on the room's edge in front of
        // it) and of the rest.
        let (doorway, rest): (Vec<Snap>, Vec<Snap>) = spots.drain(..).partition(|snap| {
            s.restore(snap);
            let r = ram(&s);
            let (x, y) = (r[0x70], r[0x84]);
            r[0x10] != 0
                && (r[0x53] != 0
                    || !(0x20..=0xD0).contains(&x)
                    || !(0x5D..=0xBD).contains(&y)
                    || ((0x70..=0x80).contains(&x) && !(0x66..=0xB4).contains(&y))
                    || ((0x85..=0x95).contains(&y) && !(0x29..=0xC7).contains(&x)))
        });
        let sample = |v: Vec<Snap>| {
            let step = (v.len() / per_state.max(1)).max(1);
            v.into_iter().step_by(step).collect::<Vec<_>>()
        };
        let mut chosen = sample(doorway);
        chosen.extend(sample(rest));
        for spot in &chosen {
            s.restore(spot);
            let view: Arc<GameView> = s.state().clone();
            let mask = compute_action_mask(&view);
            let available = view.available_actions();
            for kind in KINDS {
                let offered = view.attack_directions(kind);
                if view.has_active_enemies() && view.weapon_ready(kind) && available[kind.index()] {
                    assert_eq!(mask.directions(kind), offered, "{name} {kind:?} the mask is the plan's");
                }
                for d in directions(kind) {
                    checked += 1;
                    s.restore(spot);
                    let at = format!("{name} {kind:?} {} at {}", d.name(), describe(&s));
                    let (struck, written, stayed) = execute(&mut s, &pokes, spot, kind, d);
                    if written != 0 {
                        problems.push(format!("{at}: {written} memory writes"));
                    }
                    if offered.contains(&d) {
                        if !(struck && stayed) {
                            problems.push(format!("{at}: offered, struck {struck}, stayed {stayed}"));
                        }
                    } else if struck {
                        problems.push(format!("{at}: refused but the executor struck"));
                    } else if let Some(k) = buttons_can(&mut s, spot, kind, d) {
                        s.restore(spot);
                        let why = zelda::attack::explain(s.state(), kind, d).unwrap_err();
                        let mut line = format!("[{why:?} k={k}] {at}");
                        if std::env::var("TRIFORCE_ATTACK_DEBUG").is_ok() && why != Refusal::Band {
                            line.push_str(&trace(&mut s, spot, kind, d));
                        }
                        workable.entry(format!("{name} {why:?}")).or_insert_with(Vec::new).push(line);
                    }
                }
            }
        }
    }
    (problems, workable, checked)
}

/// The plan's model next to the game holding the strike's facing, frame by frame (debugging a refusal).
fn trace(s: &mut S, spot: &Snap, kind: ActionKind, d: Direction) -> String {
    use zelda::attack::{Link, Next, next, step};
    s.restore(spot);
    let view = s.state().clone();
    let short = |l: &Link| format!("({:02X},{:02X}) d{} o{} f{:02X} door{}", l.x, l.y, l.dir, l.offset, l.frac, l.doorway);
    let mut out = String::from("\n    model:");
    let mut l = Link::read(&view.ram()[..0x800]);
    for _ in 0..8 {
        let n = next(kind, d, &l);
        out.push_str(&format!(" {} {n:?} |", short(&l)));
        match n {
            Next::Turn(t) => match step(&view, &l, t) {
                Ok(m) => l = m,
                Err(why) => {
                    out.push_str(&format!(" step refused: {why:?}"));
                    break;
                }
            },
            _ => break,
        }
    }
    out.push_str("\n    game: ");
    for _ in 0..8 {
        s.press(direction_buttons(facing_for(d)), 1);
        out.push_str(&format!(" {} |", short(&Link::read(&ram(s)))));
    }
    out
}

/// The scan of round 21 over a sample of rooms: every offered strike is made by the controller alone (the weapon
/// faces the direction, no memory is written, Link stays in the room) and no refused one is (or it is reported).
#[test]
fn the_mask_offers_exactly_the_strikes_the_buttons_make() {
    let states = [
        "1_73s_0", "1_44w_0", "2_1es_0", "2_5es_0", "3_59e_0", "4_11s_0", "4_30e_0", "4_50e_0", "0_77c_0", "0_5bw_0",
    ];
    // TRIFORCE_ATTACK_SCAN=N: N doorway and N other positions per state from a longer walk, every refused strike
    // the buttons make listed (the round 21 scan); by default 6 + 6.
    let full: Option<usize> = std::env::var("TRIFORCE_ATTACK_SCAN").ok().and_then(|v| v.parse().ok());
    let (per_state, walk) = full.map_or((6, 300), |n| (n, 3000));
    let (problems, workable, checked) = scan(&states, per_state, walk);
    for (state, list) in &workable {
        eprintln!("{state}: {} refused strikes the buttons can make", list.len());
        for w in list.iter().take(if full.is_some() { 1000 } else { 5 }) {
            eprintln!("  {w}");
        }
    }
    eprintln!("checked {checked} strikes, {} problems", problems.len());
    assert!(problems.is_empty(), "{}", problems.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// The item sub-screen chooses every B item from every other selected item (boomerang, bombs, arrow and a candle
/// in the ring) in a dungeon and in the overworld: START, the scroll in (64 frames in a dungeon, 65 in the
/// overworld to the first active frame), one frame per step (the last one with START), the scroll out (59); then the
/// strike. The frames are reported as `menu_frames`, outside `frames`; no memory is written; the sub-screen is closed
/// and the game back in play.
#[test]
fn the_sub_screen_selects_each_b_item_from_each_other() {
    let Some((mut s, pokes)) = session() else { return };
    for (name, walk, open) in [("4_30e_0", buttons::LEFT, 64), ("0_77c_0", buttons::DOWN, 65)] {
        load(&mut s, name);
        s.nes_mut().poke(0x65B, 1); // a blue candle in slot 4 (test setup)
        s.apply_ram(&[]);
        s.press(walk, 24);
        s.press(0, 1);
        let spot = s.snapshot();
        for from in [0u8, 1, 2, 4] {
            for kind in [ActionKind::Boomerang, ActionKind::Bomb, ActionKind::Arrow] {
                let slot = zelda::submenu::slot(kind).unwrap();
                if slot == from {
                    continue;
                }
                s.restore(&spot);
                s.nes_mut().poke(0x656, from);
                s.apply_ram(&[]);
                let at = s.snapshot();
                let inventory = ram(&s);
                let (_, steps) = zelda::submenu::way(|a| inventory[a as usize], from, slot).unwrap();
                let d = s.state().attack_directions(kind)[0];
                let start = s.frame();
                let p = play(&mut s, &pokes, &at, kind, d, false);
                let r = ram(&s);
                eprintln!(
                    "{name}: {kind:?} from slot {from}: {steps} step(s), {} sub-screen frames, {} frames",
                    p.change.menu_frames, p.change.frames
                );
                assert_eq!(p.change.menu_frames, open + steps + 59, "{name} {kind:?} from {from}");
                assert_eq!((p.change.frames + p.change.menu_frames) as u64, s.frame() - start);
                assert!(p.struck && p.stayed && p.written == 0, "{name} {kind:?} {} from {from}", d.name());
                assert_eq!((r[0x656], r[0xE1], r[0x12]), (slot, 0, 5), "{name} {kind:?} from {from}");
            }
        }
        // Already selected: no sub-screen.
        s.restore(&spot);
        let d = s.state().attack_directions(ActionKind::Bomb)[0];
        let p = play(&mut s, &pokes, &spot, ActionKind::Bomb, d, true);
        assert_eq!(p.change.menu_frames, 0);
    }
}

/// Holds `pad` until `done` holds after a frame (at most `limit` frames); false if it never did.
fn hold_until(s: &mut S, pad: u8, limit: u32, mut done: impl FnMut(&[u8; 0x800]) -> bool) -> bool {
    for _ in 0..limit {
        s.press(pad, 1);
        if done(&ram(s)) {
            return true;
        }
    }
    false
}

fn link_at(s: &S) -> (u8, u8, i8, u8, u8) {
    let r = ram(s);
    (r[0x70], r[0x84], r[0x394] as i8, r[0x98], r[0x53])
}

/// A strike from the current position: it is offered, the plan's turn frames are what it took (`frames` against a
/// strike of the same weapon that needs no turn from `spot`), the weapon struck that way, no memory was written and
/// Link is still in the room. Returns (turn frames, Link after).
fn strike(s: &mut S, pokes: &AtomicU32, kind: ActionKind, d: Direction, straight: Direction, at: &str) -> (u32, (u8, u8, i8, u8, u8)) {
    let spot = s.snapshot();
    let before = link_at(s);
    let view = s.state().clone();
    let planned = zelda::attack::explain(&view, kind, d).unwrap_or_else(|why| panic!("{at}: {kind:?} {d:?} refused: {why:?}"));
    assert_eq!(zelda::attack::plan(&view, kind, straight), Some(0), "{at}: {kind:?} {straight:?} needs no turn");
    let base = play(s, pokes, &spot, kind, straight, true).change.frames;
    let p = play(s, pokes, &spot, kind, d, true);
    assert!(p.struck && p.stayed && p.written == 0, "{at}: {kind:?} {d:?} struck {} stayed {}", p.struck, p.stayed);
    let turn = p.change.frames - base;
    assert_eq!(turn, planned, "{at}: {kind:?} {d:?} turn frames");
    let after = link_at(s);
    eprintln!("{at}: {kind:?} {}: {turn} turn frame(s), {} frames in all; Link {before:02X?} -> {after:02X?}", d.name(), p.change.frames);
    s.restore(&spot);
    (turn, after)
}

/// Turn and strike, situation by situation, in the real game: the sword turns on its press (no frame, no step), a
/// bomb or arrow turns first (one frame, a step of 1-2 pixels when the way is open, none against a wall), off the
/// grid the game walks Link to the nearer grid point first, a border band that drops the button makes a turn first
/// or refuses the strike (a walk out of it would be a MOVE), a doorway facing into the room walks Link back onto
/// the room's edge, facing out of it refuses sideways strikes, the laid ladder turns Link in place, and a weapon
/// still out masks only itself.
#[test]
fn turn_and_strike_in_every_situation() {
    use ActionKind::{Arrow, Boomerang, Sword};
    use Direction::{E, N, S, W};
    let Some((mut s, pokes)) = session() else { return };
    // 1:0x73 from S: a grid point inside the room, facing N.
    load(&mut s, "1_73s_0");
    assert!(hold_until(&mut s, buttons::UP, 60, |r| r[0x84] <= 0xA5 && r[0x394] == 0));
    s.press(0, 1);
    let grid = s.snapshot();
    let (x0, y0, ..) = link_at(&s);
    let (turn, after) = strike(&mut s, &pokes, Sword, E, N, "grid");
    assert_eq!((turn, after.0, after.1), (0, x0, y0), "the sword turns on its press without a step");
    let (turn, after) = strike(&mut s, &pokes, Arrow, E, N, "grid");
    assert!(turn == 1 && (1..=2).contains(&(after.0 - x0)) && after.1 == y0, "an arrow turns first, a step E");
    // Off the grid across the strike: 4+ pixels out the game goes on to the next grid point, under 4 back.
    s.restore(&grid);
    s.press(buttons::UP, 3);
    let (_, _, offset, ..) = link_at(&s);
    assert!(offset <= -4, "{offset}");
    let (turn, after) = strike(&mut s, &pokes, Sword, E, N, "off grid, ahead");
    assert!((1..=3).contains(&turn) && after.1 == y0 - 8, "{turn} {after:?}");
    s.restore(&grid);
    s.press(buttons::UP, 1);
    let (turn, after) = strike(&mut s, &pokes, Sword, W, N, "off grid, back");
    assert!((1..=2).contains(&turn) && after.1 == y0, "{turn} {after:?}");
    // Against a wall: the turn toward it doesn't move Link (walk W until blocked; a swing N then turns him N in
    // place).
    s.restore(&grid);
    let mut last = link_at(&s).0;
    assert!(hold_until(&mut s, buttons::LEFT, 120, |r| {
        let blocked = r[0x70] == last && r[0x394] == 0;
        last = r[0x70];
        blocked
    }));
    s.act(Action::new(Sword, N, 0).unwrap()).unwrap();
    let (wx, wy, ..) = link_at(&s);
    let (turn, after) = strike(&mut s, &pokes, Arrow, W, N, "wall");
    assert_eq!((turn, after.0, after.1), (1, wx, wy), "turned against the wall in place");

    // A weapon still out masks only itself: the boomerang in flight, the sword still strikes.
    s.restore(&grid);
    let p = play(&mut s, &pokes, &grid, Boomerang, N, true);
    assert!(p.struck);
    if s.nes().ram()[0xBB] != 0 {
        let view = s.state();
        assert!(!view.weapon_ready(Boomerang) && !s.mask().kind[Boomerang.index()], "the boomerang is still out");
        assert!(view.weapon_ready(Sword) && view.attack_directions(Sword).contains(&E));
    }

    // The overworld W edge facing E (0x5B from W): the band behind Link drops A and B facing E, so E is refused;
    // N and S turn first (one frame).
    load(&mut s, "0_5bw_0");
    let view = s.state().clone();
    assert!(view.link.position.x < 7, "{:?}", link_at(&s));
    assert_eq!(zelda::attack::explain(&view, Sword, E), Err(Refusal::Band));
    assert_eq!(zelda::attack::explain(&view, Arrow, E), Err(Refusal::Band));
    let mut turned = 0;
    for d in [N, S] {
        if let Some(frames) = zelda::attack::plan(&view, Sword, d) {
            let spot = s.snapshot();
            let p = play(&mut s, &pokes, &spot, Sword, d, true);
            assert!(p.struck && p.stayed && p.written == 0, "0x5B edge: SWORD {d:?}");
            assert_eq!(frames, 1);
            turned += 1;
            s.restore(&spot);
        }
    }
    assert!(turned > 0);

    // 4:0x11's S doorway (round 20's wall walk past the open bomb wall): on the entry spot DoorwayDir is set and
    // Link faces N, into the room; a step S puts him in the doorway with DoorwayDir clear (78,BF) facing S. From
    // both a sideways strike is the game's: it walks him back onto the room's edge and turns him there.
    load(&mut s, "4_11s_0");
    for step in [0, 1] {
        s.press(buttons::DOWN, step);
        let at = link_at(&s);
        let spot = s.snapshot();
        let p = play(&mut s, &pokes, &spot, Sword, E, true);
        assert!(p.struck && p.stayed && p.written == 0, "4:0x11 {at:02X?}");
        let (_, y, _, _, door) = link_at(&s);
        assert!(y <= 0xBD && door == 0, "back in the room: {:02X?}", link_at(&s));
        s.restore(&spot);
    }
    // Deep in 1:0x73's S doorway facing S (out of the room) a sideways strike is refused, and the buttons can't
    // make it either: the forced step walks Link out through the door.
    load(&mut s, "1_73s_0");
    assert!(hold_until(&mut s, buttons::DOWN, 30, |r| r[0x53] != 0 && r[0x84] >= 0xC5 && r[0x394] == 0));
    let deep = s.snapshot();
    assert_eq!(zelda::attack::explain(s.state(), Sword, E), Err(Refusal::Doorway), "{:02X?}", link_at(&s));
    assert_eq!(buttons_can(&mut s, &deep, Sword, E), None, "the game can't either");

    // 4:0x30 from E: onto the ladder over the water row; on it Link turns in place.
    load(&mut s, "4_30e_0");
    assert!(hold_until(&mut s, buttons::LEFT, 120, |r| r[0x70] == 0x78 && r[0x394] == 0));
    let on_ladder = |r: &[u8; 0x800]| {
        let slot = r[0x64] as usize;
        slot != 0 && r[0x394] == 0 && r[0x70] == r[0x70 + slot] && r[0x84].wrapping_add(3) == r[0x84 + slot]
    };
    assert!(hold_until(&mut s, buttons::UP, 120, on_ladder), "{:?}", link_at(&s));
    s.press(0, 1);
    let (lx, ly, ..) = link_at(&s);
    let (turn, after) = strike(&mut s, &pokes, Sword, E, N, "ladder");
    assert_eq!((turn, after.0, after.1), (0, lx, ly));
    let (turn, after) = strike(&mut s, &pokes, Arrow, E, N, "ladder");
    assert_eq!((turn, after.0, after.1), (1, lx, ly), "turned on the ladder in place");
}
