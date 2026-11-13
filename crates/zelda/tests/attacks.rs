//! Attacks by buttons in the real game (round 21 movement cleanup; `zelda::attack`, `zelda::submenu`): the mask
//! offers exactly the strikes the controller can make, the executor makes them without writing memory, and the item
//! sub-screen selects every B item from every other one; with the clips (`zelda::clips`): attacks at each clip's
//! end, a clip right after an attack, and the turn and MOVE models of a laid ladder against the ladder clip's frames.
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
            checked += check_spot(&mut s, &pokes, name, spot, true, &mut problems, &mut workable);
        }
    }
    (problems, workable, checked)
}

/// Every attack from `spot` against the game: an offered one is struck the way asked by the controller alone with
/// Link staying in the room, a refused one isn't struck by the executor; a refused strike that plain buttons can
/// make is listed in `workable` under its refusal. `mask_is_plan` also checks the action mask offers the plan's
/// directions where the weapon is legal. Returns the strikes checked.
fn check_spot(
    s: &mut S,
    pokes: &AtomicU32,
    name: &str,
    spot: &Snap,
    mask_is_plan: bool,
    problems: &mut Vec<String>,
    workable: &mut BTreeMap<String, Vec<String>>,
) -> usize {
    let mut checked = 0;
    s.restore(spot);
    let view: Arc<GameView> = s.state().clone();
    let mask = compute_action_mask(&view);
    let available = view.available_actions();
    for kind in KINDS {
        let offered = view.attack_directions(kind);
        if mask_is_plan && view.has_active_enemies() && view.weapon_ready(kind) && available[kind.index()] {
            assert_eq!(mask.directions(kind), offered, "{name} {kind:?} the mask is the plan's");
        }
        for d in directions(kind) {
            checked += 1;
            s.restore(spot);
            let at = format!("{name} {kind:?} {} at {}", d.name(), describe(s));
            let (struck, written, stayed) = execute(s, pokes, spot, kind, d);
            if written != 0 {
                problems.push(format!("{at}: {written} memory writes"));
            }
            if offered.contains(&d) {
                if !(struck && stayed) {
                    problems.push(format!("{at}: offered, struck {struck}, stayed {stayed}"));
                }
            } else if struck {
                problems.push(format!("{at}: refused but the executor struck"));
            } else if let Some(k) = buttons_can(s, spot, kind, d) {
                s.restore(spot);
                let why = zelda::attack::explain(s.state(), kind, d).unwrap_err();
                let mut line = format!("[{why:?} k={k}] {at}");
                if std::env::var("TRIFORCE_ATTACK_DEBUG").is_ok() && why != Refusal::Band {
                    line.push_str(&trace(s, spot, kind, d));
                }
                workable.entry(format!("{name} {why:?}")).or_default().push(line);
            }
        }
    }
    checked
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
        "1_73s_0", "1_44w_0", "2_1es_0", "2_5es_0", "3_59e_0", "4_11s_0", "4_21s_0", "4_50e_0", "0_77c_0", "0_5bw_0",
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
    for (name, walk, open) in [("4_31e_0", buttons::LEFT, 64), ("0_77c_0", buttons::DOWN, 65)] {
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

    // 4:0x30 from E (walked into from 0x31's W door, as round 21's route no longer enters it): onto the ladder over
    // the water row; on it Link turns in place.
    load(&mut s, "4_31e_0");
    s.apply_ram(&[(zelda::Var::DoorwayDir, 0)]);
    for (addr, value) in [(0x70, 0x20), (0x84, 0x8D), (0x394, 0)] {
        s.nes_mut().poke(addr, value); // Link beside 0x31's W door (test setup)
    }
    s.apply_ram(&[]);
    for _ in 0..8 {
        if s.state().location() == 0x30 {
            break;
        }
        s.act(Action::parse("MOVE W 8").unwrap()).unwrap();
    }
    assert_eq!((s.state().location(), ram(&s)[0x12]), (0x30, 5), "{:?}", link_at(&s));
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

/// The end state of each kind of clip, clips on: the wrap W and E on the overworld start screen's row (X $F0 and
/// $00; a sword given by test setup), the block clip N under 1:$22's east block (8 px into it, ($A0, $8D)) and in
/// 4:$50 (($B0, $8D) to ($B0, $7D)), and the ladder clip N off 4:$21's island (X, $55) on three of its columns.
/// Each with the clip's start (in the zone).
fn clip_ends(s: &mut S) -> Vec<(String, Snap, Snap)> {
    use Direction::{E, N, W};
    let mut out = Vec::new();
    let mut clip = |s: &mut S, label: String, d: Direction, end: (u8, u8)| {
        assert!(zelda::clips::skip_available(s.state(), d), "{label}: not offered at {}", describe(s));
        let start = s.snapshot();
        let c = s.act(Action::parse(&format!("CLIP {}", d.name())).unwrap()).unwrap();
        let outcome = c.clip.expect("a clip outcome");
        assert!(outcome.completed(), "{label}: {outcome:?} at {}", describe(s));
        assert_eq!((link_at(s).0, link_at(s).1, link_at(s).2), (end.0, end.1, 0), "{label}: {}", describe(s));
        out.push((label, start, s.snapshot()));
    };
    for (d, pad, end_x) in [(W, buttons::LEFT, 0xF0), (E, buttons::RIGHT, 0x00)] {
        load(s, "start");
        s.nes_mut().poke(0x657, 1); // the wooden sword (test setup)
        s.apply_ram(&[]);
        s.set_clips(true);
        let zone = |x: u8| if d == W { x <= 0x20 } else { x >= 0xD0 };
        assert!(hold_until(s, pad, 120, |r| zone(r[0x70]) && r[0x394] == 0));
        clip(s, format!("wrap {}", d.name()), d, (end_x, 0x8D));
    }
    load(s, "1_22e_0");
    s.set_clips(true);
    assert!(hold_until(s, buttons::LEFT, 120, |r| r[0x70] == 0xB0 && r[0x394] == 0));
    assert!(hold_until(s, buttons::DOWN, 120, |r| r[0x84] == 0xAD && r[0x394] == 0));
    assert!(hold_until(s, buttons::LEFT, 120, |r| r[0x70] == 0xA0 && r[0x394] == 0));
    assert!(hold_until(s, buttons::UP, 120, |r| r[0x84] == 0xA5 && r[0x394] == 0));
    clip(s, "block N under 1:$22's east block".into(), N, (0xA0, 0x8D));
    // 4:$50 (the Vires' room), where round 21's experiment clips: N from ($B0, $8D) into the block above.
    load(s, "4_50e_0");
    s.apply_ram(&[(zelda::Var::DoorwayDir, 0)]);
    for (addr, value) in [(0x70, 0xB0), (0x84, 0x8D), (0x394, 0)] {
        s.nes_mut().poke(addr, value); // test setup
    }
    s.apply_ram(&[]);
    s.set_clips(true);
    clip(s, "block N in 4:$50".into(), N, (0xB0, 0x7D));
    for x in [0x50u8, 0x78, 0xA0] {
        load(s, LADDER_21);
        s.apply_ram(&[(zelda::Var::DoorwayDir, 0)]);
        for (addr, value) in [(0x70, x), (0x84, 0x9D), (0x394, 0)] {
            s.nes_mut().poke(addr, value); // Link on the island (test setup)
        }
        s.apply_ram(&[]);
        s.set_clips(true);
        assert!(s.state().ladder(), "{}", describe(s));
        assert!(hold_until(s, buttons::UP, 30, |r| r[0x84] == 0x8D && r[0x394] == 0));
        clip(s, format!("ladder N in 4:$21 from X {x:02X}"), N, (x, 0x55));
    }
    out
}

/// 4:$21 with the ladder, Link entering where the route does.
const LADDER_21: &str = "4_21s_0";

/// Attacks right after each kind of clip, from its end and from one and three frames of each direction after it:
/// every strike offered is made the way asked by the controller alone with Link staying in the room, and every one
/// refused is one the buttons can't make, or only after a walk out of a border band (a MOVE first: `Band`).
#[test]
fn attacks_right_after_each_clip() {
    let Some((mut s, pokes)) = session() else { return };
    let (mut problems, mut workable, mut checked) = (Vec::new(), BTreeMap::new(), 0);
    for (label, _, end) in clip_ends(&mut s) {
        s.restore(&end);
        let room = room_of(&s);
        let offers: Vec<String> = KINDS
            .iter()
            .map(|&k| format!("{k:?} {:?}", s.state().attack_directions(k).iter().map(|d| d.name()).collect::<Vec<_>>()))
            .collect();
        eprintln!("{label}: at {}: {}", describe(&s), offers.join(", "));
        let mut spots = vec![end.clone()];
        for pad in [buttons::UP, buttons::DOWN, buttons::LEFT, buttons::RIGHT] {
            for frames in [1, 3] {
                s.restore(&end);
                s.press(pad, frames);
                if in_play(&s, room) && ram(&s)[0xAC] == 0 {
                    spots.push(s.snapshot());
                }
            }
        }
        for spot in &spots {
            checked += check_spot(&mut s, &pokes, &label, spot, true, &mut problems, &mut workable);
        }
    }
    let mut band = 0;
    for (key, list) in &workable {
        if key.ends_with("Band") {
            band += list.len();
        } else {
            problems.extend(list.iter().cloned());
        }
    }
    eprintln!("checked {checked} strikes after clips; {band} refused by a border band the buttons leave by walking");
    assert!(problems.is_empty(), "{}", problems.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// A B item that is not the selected one, right at each clip's end: the action opens the sub-screen (the boomerang
/// selected, a BOMB asked), selects the bombs, closes it and places the bomb the way asked; the frames are
/// `menu_frames`, no memory is written and Link stays in the room.
#[test]
fn a_b_item_switch_at_each_clips_end() {
    let Some((mut s, pokes)) = session() else { return };
    let boomerang = zelda::submenu::slot(ActionKind::Boomerang).unwrap();
    let mut switched = 0;
    for (label, _, end) in clip_ends(&mut s) {
        s.restore(&end);
        s.nes_mut().poke(0x656, boomerang); // test setup: the boomerang selected
        s.apply_ram(&[]);
        let spot = s.snapshot();
        for d in s.state().attack_directions(ActionKind::Bomb) {
            let p = play(&mut s, &pokes, &spot, ActionKind::Bomb, d, false);
            assert!(p.struck && p.stayed && p.written == 0, "{label}: BOMB {} struck {} stayed {}", d.name(), p.struck, p.stayed);
            assert!(p.change.menu_frames > 100, "{label}: BOMB {}: {} sub-screen frames", d.name(), p.change.menu_frames);
            assert_eq!(ram(&s)[0x656], zelda::submenu::slot(ActionKind::Bomb).unwrap());
            switched += 1;
        }
    }
    assert!(switched >= 6, "{switched} switches");
}

/// A CLIP right after an attack, from a start of each clip's zone: with the sword out (A pressed on the frame
/// before) and in the catch pose of a returning boomerang ($AC != 0) the skip is either not offered or completes.
#[test]
fn a_clip_right_after_an_attack() {
    let Some((mut s, _)) = session() else { return };
    let (mut refused, mut done) = (0, 0);
    for (label, start, end) in clip_ends(&mut s) {
        s.restore(&end);
        let want = (link_at(&s).0, link_at(&s).1);
        let d = s.state().full_location();
        s.restore(&start);
        let heading = [Direction::N, Direction::S, Direction::W, Direction::E]
            .into_iter()
            .find(|&h| zelda::clips::skip_available(s.state(), h))
            .unwrap();
        let mut posed = Vec::new();
        // The sword out.
        s.press(buttons::A, 1);
        if ram(&s)[0xB9] != 0 {
            posed.push(("sword out", s.snapshot()));
        }
        // The catch pose: a boomerang thrown away from the clip, waited for.
        s.restore(&start);
        let away = heading.opposite();
        if s.state().attack_directions(ActionKind::Boomerang).contains(&away) {
            select(&mut s, ActionKind::Boomerang);
            s.act(Action::new(ActionKind::Boomerang, away, 0).unwrap()).unwrap();
            if hold_until(&mut s, 0, 180, |r| r[0xAC] != 0) {
                posed.push(("catch pose", s.snapshot()));
            }
        }
        assert!(!posed.is_empty(), "{label}: no pose reached");
        for (pose, snap) in posed {
            s.restore(&snap);
            let at = format!("{label}, {pose}: {} $AC={:02X} $B9={:02X}", describe(&s), ram(&s)[0xAC], ram(&s)[0xB9]);
            if !zelda::clips::skip_available(s.state(), heading) {
                assert!(!s.mask().is_allowed(&Action::new(ActionKind::Clip, heading, 0).unwrap()), "{at}");
                refused += 1;
                continue;
            }
            let c = s.act(Action::new(ActionKind::Clip, heading, 0).unwrap()).unwrap();
            let outcome = c.clip.expect("a clip outcome");
            assert!(outcome.completed(), "{at}: offered but {outcome:?}, now {}", describe(&s));
            assert_eq!((s.state().full_location(), link_at(&s).0, link_at(&s).1), (d, want.0, want.1), "{at}");
            done += 1;
        }
    }
    eprintln!("a CLIP right after an attack: {refused} refused, {done} completed");
}

/// 4:$21's N bomb wall after the ladder clip (end X $78, Y $55, 8 px into the top wall). BOMB N there is offered and
/// the buttons place the bomb N, but it opens nothing: the bomb lies at Y $45, `$18` from the up wall's hotspot
/// ($78, $5D), and the wall test wants less (asm:Bomb_TryOpenWall, asm:Bomb_WallHotspotsY; mechanics.md "Bomb
/// walls"). From the N door spot ($78, $5D) a step S of the clip's end, BOMB N opens it and MOVE N leaves by it into
/// $11 (the route's step 0x21 S -> N).
#[test]
fn bomb_n_after_the_ladder_clip() {
    let Some((mut s, pokes)) = session() else { return };
    let (label, _, end) = clip_ends(&mut s).into_iter().find(|(label, ..)| label.ends_with("X 78")).unwrap();
    s.restore(&end);
    assert!(!s.state().is_bomb_door_open(Direction::N), "{label}: the wall is open already");
    assert!(s.state().attack_directions(ActionKind::Bomb).contains(&Direction::N), "{label}");
    let p = play(&mut s, &pokes, &end, ActionKind::Bomb, Direction::N, true);
    assert!(p.struck && p.stayed && p.written == 0, "{label}: BOMB N struck {} stayed {}", p.struck, p.stayed);
    let r = ram(&s);
    assert!((0..2).any(|i| r[0xBC + i] != 0 && r[0x84 + 0x10 + i] == 0x45), "{label}: bomb Y {:02X?}", &r[0x94..0x96]);
    s.press(0, 120); // the fuse and the blast
    assert!(!s.state().is_bomb_door_open(Direction::N), "{label}: a bomb from Y $55 opened the wall");

    s.restore(&end);
    assert!(hold_until(&mut s, buttons::DOWN, 30, |r| r[0x84] == 0x5D && r[0x394] == 0), "{}", describe(&s));
    let spot = s.snapshot();
    let p = play(&mut s, &pokes, &spot, ActionKind::Bomb, Direction::N, true);
    assert!(p.struck && p.stayed && p.written == 0, "{label}: BOMB N from $5D");
    s.press(0, 120);
    assert!(s.state().is_bomb_door_open(Direction::N), "{label}: the N wall stayed shut at {}", describe(&s));
    let room = s.state().full_location();
    for _ in 0..4 {
        if s.state().full_location() != room {
            break;
        }
        s.act(Action::parse("MOVE N 16").unwrap()).unwrap();
    }
    assert_eq!(s.state().full_location(), zelda::MapLocation::new(4, 0x11, false), "{}", describe(&s));
}

/// Link stands on the laid ladder's line less than `$10` pixels from it (asm:Ladder_CheckAndDraw's test).
fn on_laid_ladder(r: &[u8; 0x800]) -> bool {
    let slot = r[0x64] as usize;
    if slot == 0 || slot >= 12 {
        return false;
    }
    let (x, y, lx, ly) = (r[0x70], r[0x84], r[0x70 + slot], r[0x84 + slot]);
    if r[0x98 + slot] & 0x0C != 0 {
        x == lx && (y.wrapping_add(3).wrapping_sub(ly) as i8).unsigned_abs() < 0x10
    } else {
        y.wrapping_add(3) == ly && (x.wrapping_sub(lx) as i8).unsigned_abs() < 0x10
    }
}

/// The two models of a laid ladder against the game on every frame of the ladder clips of 4:$21 (N off the island
/// on every column, W and E on its three rows, S from the top row) where Link is on the laid ladder: the turn model
/// (`attack::step`, asm:Ladder_CheckAndDraw frame by frame) puts Link where one frame of each direction does, but
/// for its unmodelled upward case E (`Refusal::Ladder`), and on a grid point the MOVE model
/// (`GameView::can_link_move`: `ladder_move`, `lays_ladder`) allows exactly the directions that move Link that way
/// within four frames.
#[test]
fn the_ladder_models_agree_on_the_ladder_clip_traces() {
    use Direction::{E, N, S, W};
    let Some((mut s, _)) = session() else { return };
    let mut starts = Vec::new();
    for x in (0x50..=0xA0).step_by(8) {
        starts.push((x, 0x8D, N));
        starts.push((x, 0x5D, S));
    }
    for row in [0x8D, 0x95, 0x9D] {
        starts.push((0x50, row, W));
        starts.push((0xA0, row, E));
    }
    let (mut frames, mut moves, mut case_e) = (0, 0, 0);
    for (x, y, heading) in starts {
        load(&mut s, LADDER_21);
        s.apply_ram(&[(zelda::Var::DoorwayDir, 0)]);
        for (addr, value) in [(0x70, x), (0x84, y), (0x394, 0)] {
            s.nes_mut().poke(addr, value); // Link put at rest (test setup)
        }
        s.apply_ram(&[]);
        s.set_clips(true);
        let label = format!("CLIP {} from ({x:02X},{y:02X})", heading.name());
        assert!(zelda::clips::skip_available(s.state(), heading), "{label}");
        let trace = Arc::new(Mutex::new(Vec::new()));
        let sink = trace.clone();
        s.set_frame_sink(Some(Box::new(move |n: &Poked| sink.lock().push(n.snapshot()))));
        let c = s.act(Action::new(ActionKind::Clip, heading, 0).unwrap()).unwrap();
        s.set_frame_sink(None);
        assert!(c.clip.is_some_and(|o| o.completed()), "{label}: {:?}", c.clip);
        for state in trace.lock().iter() {
            s.nes_mut().restore(state);
            s.apply_ram(&[]);
            let r = ram(&s);
            if r[0x12] != 5 || r[0xAC] != 0 || !on_laid_ladder(&r) {
                continue;
            }
            frames += 1;
            let view = s.state().clone();
            let link = zelda::attack::Link::read(&r);
            let at = format!("{label}: at {} ladder {:02X?}", describe(&s), (r[0x64], r[0x70 + r[0x64] as usize], r[0x84 + r[0x64] as usize], r[0x98 + r[0x64] as usize]));
            let here = s.snapshot();
            for d in [N, S, W, E] {
                s.restore(&here);
                s.press(direction_buttons(d), 1);
                let game = (link_at(&s).0, link_at(&s).1);
                match zelda::attack::step(&view, &link, d) {
                    Ok(n) => assert_eq!((n.x, n.y), game, "{at}: one frame {}: the turn model", d.name()),
                    Err(Refusal::Ladder) => case_e += 1,
                    Err(why) => panic!("{at}: one frame {}: the turn model refuses ({why:?})", d.name()),
                }
                if r[0x394] != 0 {
                    continue;
                }
                moves += 1;
                s.restore(&here);
                let start = (r[0x70] as i32, r[0x84] as i32);
                let mut moved = false;
                for _ in 0..4 {
                    s.press(direction_buttons(d), 1);
                    let (nx, ny) = (link_at(&s).0 as i32, link_at(&s).1 as i32);
                    let along = match d {
                        N => start.1 - ny,
                        S => ny - start.1,
                        W => start.0 - nx,
                        _ => nx - start.0,
                    };
                    moved |= along > 0;
                }
                assert_eq!(view.can_link_move(d), moved, "{at}: MOVE {}: the MOVE model", d.name());
            }
        }
    }
    eprintln!("{frames} frames on the laid ladder, {moves} MOVE checks on a grid point, {case_e} upward case E frames");
    assert!(frames > 100 && moves > 20, "{frames} {moves}");
}

/// A boomerang caught around a CLIP's decision, from a start of each clip's zone (round 21's 4:$50 experiment saw
/// `CLIP N` stall when the catch pose, `$AC` = `$20`/`$21`/`$31`, began during the macro): the boomerang thrown
/// away from the clip and Link left standing 0..k frames, k from four frames before the catch to two into it. With
/// the pose on at the decision CLIP is not offered; otherwise it is, and it completes at the clip's end, the frames
/// of a pose that starts during the macro waited out.
#[test]
fn a_boomerang_caught_around_a_clip() {
    let Some((mut s, _)) = session() else { return };
    let (mut refused, mut caught_in_macro, mut done) = (0, 0, 0);
    for (label, start, end) in clip_ends(&mut s) {
        s.restore(&end);
        let (room, want) = (s.state().full_location(), (link_at(&s).0, link_at(&s).1));
        s.restore(&start);
        let heading = [Direction::N, Direction::S, Direction::W, Direction::E]
            .into_iter()
            .find(|&h| zelda::clips::skip_available(s.state(), h))
            .unwrap();
        let away = heading.opposite();
        if !s.state().attack_directions(ActionKind::Boomerang).contains(&away) {
            continue;
        }
        select(&mut s, ActionKind::Boomerang);
        s.act(Action::new(ActionKind::Boomerang, away, 0).unwrap()).unwrap();
        let thrown = s.snapshot();
        let mut catch = None;
        for f in 0..180u32 {
            if ram(&s)[0xAC] != 0 {
                catch = Some(f);
                break;
            }
            s.press(0, 1);
        }
        // Not every catch puts Link in the pose (none within 180 frames from 4:$21's X $78 and $A0).
        let Some(catch) = catch else {
            eprintln!("{label}: no catch pose within 180 frames");
            continue;
        };
        for k in catch.saturating_sub(4)..=catch + 2 {
            s.restore(&thrown);
            s.press(0, k);
            let at = format!("{label}: {k} frames after the throw (catch at {catch}): {} $AC={:02X}", describe(&s), ram(&s)[0xAC]);
            if !zelda::clips::skip_available(s.state(), heading) {
                assert!(ram(&s)[0xAC] != 0, "{at}: not offered with Link free");
                refused += 1;
                continue;
            }
            caught_in_macro += usize::from(k < catch);
            let c = s.act(Action::new(ActionKind::Clip, heading, 0).unwrap()).unwrap();
            let outcome = c.clip.expect("a clip outcome");
            assert!(outcome.completed(), "{at}: {outcome:?}, now {}", describe(&s));
            assert_eq!((s.state().full_location(), link_at(&s).0, link_at(&s).1), (room, want.0, want.1), "{at}");
            done += 1;
        }
    }
    eprintln!("boomerang around a CLIP: {refused} refused in the pose, {done} completed ({caught_in_macro} caught during the macro)");
    assert!(refused > 0 && caught_in_macro > 0, "{refused} {caught_in_macro}");
}

/// Wherever a clip can be cut short (a hit, a stand-still, a miss or the frame cap leave Link where he stands: every
/// frame of each clip of [`clip_ends`], from its start to its end, partly inside a block, over the water on a
/// ladder or past the screen's bound), the masks are the game's: the MOVE mask allows exactly the directions that
/// move Link (or change the room) within 8 frames of holding them, and on every third such frame every attack
/// offered is struck the way asked and every one refused is one the buttons can't make (but after a walk out of a
/// border band).
#[test]
fn the_masks_are_the_games_wherever_a_clip_is_cut() {
    let Some((mut s, pokes)) = session() else { return };
    let (mut problems, mut workable) = (Vec::new(), BTreeMap::new());
    let (mut frames, mut inside) = (0, 0);
    for (label, start, _) in clip_ends(&mut s) {
        s.restore(&start);
        let heading = [Direction::N, Direction::S, Direction::W, Direction::E]
            .into_iter()
            .find(|&h| zelda::clips::skip_available(s.state(), h))
            .unwrap();
        let room = room_of(&s);
        let trace = Arc::new(Mutex::new(Vec::new()));
        let sink = trace.clone();
        s.set_frame_sink(Some(Box::new(move |n: &Poked| sink.lock().push(n.snapshot()))));
        s.act(Action::new(ActionKind::Clip, heading, 0).unwrap()).unwrap();
        s.set_frame_sink(None);
        for (i, state) in trace.lock().iter().enumerate() {
            s.nes_mut().restore(state);
            s.apply_ram(&[]);
            let r = ram(&s);
            if !in_play(&s, room) || r[0xAC] != 0 {
                continue;
            }
            frames += 1;
            inside += usize::from(r[0x394] != 0);
            let at = format!("{label}, frame {i} of the clip: {}", describe(&s));
            let here = s.snapshot();
            let mask = *s.mask();
            for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
                s.restore(&here);
                let start = link_at(&s);
                s.press(direction_buttons(d), 8);
                let moved = link_at(&s).0 != start.0 || link_at(&s).1 != start.1 || !in_play(&s, room);
                let legal = mask.is_allowed(&Action::new(ActionKind::Move, d, 8).unwrap());
                if legal != moved {
                    problems.push(format!("{at}: MOVE {} legal {legal}, the game moved {moved}", d.name()));
                }
            }
            if i % 3 == 0 {
                check_spot(&mut s, &pokes, &at, &here, true, &mut problems, &mut workable);
            }
        }
    }
    for (key, list) in &workable {
        if !key.ends_with("Band") {
            problems.extend(list.iter().cloned());
        }
    }
    eprintln!("{frames} frames of clips checked, {inside} of them off the grid");
    assert!(problems.is_empty(), "{}", problems.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}
