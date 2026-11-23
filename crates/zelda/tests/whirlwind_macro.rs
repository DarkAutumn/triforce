//! Round 40: the engine macros of the fresh generation (routes.yaml `macros:`, docs/routes/d3first.md "The engine's
//! macros"; decisions 2026-11-23 "The fresh generation's design calls"). The whirlwinds out of Level 5 (0:0x0b) and
//! Level 6 (0:0x22) as one RECORDER that turns, plays and waits for the pickup (`Warp::Whirlwind`), and the dock warps
//! unaided (`Warp::Dock { unaided: true }`): both masked by Link's position only, never by a monster or a shot, and
//! both off on `d3first` as landed (today's team plays as it was trained).
//!
//! SETUP (RAM writes): monsters and shots are removed (`tidy`) or placed to compare the masks with and without them,
//! and removed before the plays that must land (the live-monster numbers are docs/experiments/round40.md's).
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::enums::ActionKind;
use zelda::game::GameView;
use zelda::game_map::{GAME_YAML, Macros, ROUTES_YAML};
use zelda::objectives::GameCompletion;
use zelda::whirlwind::Warp;
use zelda::{Action, GameMap, LoadOptions, MapLocation, Session};

type S = Session<Counted>;

const ROUTE: &str = "d3first";
/// routes.yaml's `macros:` entry for `d3first` as landed (off), and the fresh generation's (on).
const OFF: &str = "macros:\n  d3first:\n    whirlwinds: []\n    unaided_docks: false\n";
const ON: &str = "macros:\n  d3first:\n    whirlwinds: [{level: 0, room: '0x0b'}, {level: 0, room: '0x22'}]\n    \
                  unaided_docks: true\n";

/// The map with `d3first`'s macros on.
fn map_on() -> Arc<GameMap> {
    assert!(ROUTES_YAML.contains(OFF), "routes.yaml's d3first macros entry is the landed one");
    Arc::new(GameMap::from_yaml(GAME_YAML, &ROUTES_YAML.replacen(OFF, ON, 1)).unwrap())
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

const OBJ_TYPE: u16 = 0x34F;
const OBJ_X: u16 = 0x70;
const OBJ_Y: u16 = 0x84;
/// Object types [`tidy`] keeps (as `dock_warp.rs`): the dock, the ladder, the rock wall, the whirlwind, blocks.
const KEEP: [u8; 9] = [0x61, 0x5F, 0x63, 0x2E, 0x68, 0x65, 0x64, 0x5E, 0x33];

/// A session on `state` with the route's completion at the step leaving `room` by the whirlwind, the env's per-step
/// wiring done once ([`Walk::wire`]).
struct Walk {
    s: S,
    gc: GameCompletion,
    prev: Option<Arc<GameView>>,
}

impl Walk {
    fn start(map: &Arc<GameMap>, state: &str, room: u8) -> Option<Walk> {
        let core = open_core()?;
        let mut s = Session::new(Counted(core, 0), map.clone());
        s.load(&load_named_state(state), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        s.set_clips(true);
        let step = map.routes[ROUTE].iter().position(|st| st.location() == ow(room) && st.exit == "recorder").unwrap();
        let gc = GameCompletion::new(map.clone(), ROUTE, Some(step)).unwrap();
        let mut w = Walk { s, gc, prev: None };
        w.wire();
        Some(w)
    }

    /// What the env sets before every decision (`env::chain`'s `update_disabled_kinds`): the objective, the recorder's
    /// facings and the warp.
    fn wire(&mut self) {
        let state = self.s.state().clone();
        self.gc.objectives(self.prev.as_deref(), &state);
        self.s.set_recorder_facings(Some(self.gc.recorder_facings(&state)));
        self.s.set_warp(self.gc.warp(&state));
        self.prev = Some(state);
    }

    fn act(&mut self, text: &str) {
        self.s.act(Action::parse(text).unwrap()).unwrap();
        self.wire();
    }

    fn allowed(&self, text: &str) -> bool {
        self.s.mask().is_allowed(&Action::parse(text).unwrap())
    }

    /// Removes the room's monsters and shots.
    fn tidy(&mut self) {
        for slot in 1..=11u16 {
            let t = self.s.state().ram()[(OBJ_TYPE + slot) as usize];
            if t != 0 && !KEEP.contains(&t) {
                zelda::Nes::poke(self.s.nes_mut(), OBJ_TYPE + slot, 0);
            }
        }
        self.s.apply_ram(&[]);
    }

    fn place(&mut self, kind: u8, x: u8, y: u8) {
        for (address, value) in [(OBJ_TYPE + 1, kind), (OBJ_X + 1, x), (OBJ_Y + 1, y)] {
            zelda::Nes::poke(self.s.nes_mut(), address, value);
        }
        self.s.apply_ram(&[]);
    }

    /// The part of the mask a warp sets: RECORDER, CLIP and MOVE with their directions.
    fn warp_part(&self) -> Vec<bool> {
        let m = self.s.mask();
        [ActionKind::Recorder, ActionKind::Clip, ActionKind::Move]
            .into_iter()
            .flat_map(|k| std::iter::once(m.kind[k.index()]).chain(m.direction[k.index()]))
            .collect()
    }

    /// [`Walk::warp_part`] as Link stands, and with the room's monsters and shots removed (the state put back after).
    fn with_and_without_monsters(&mut self) -> (Vec<bool>, Vec<bool>) {
        let live = self.warp_part();
        let snapshot = self.s.snapshot();
        self.tidy();
        let none = self.warp_part();
        self.s.restore(&snapshot);
        self.wire();
        (live, none)
    }

    /// Waits until Link is in play outside `room`; his room and X.
    fn landing(&mut self, room: u8) -> (MapLocation, i32) {
        for _ in 0..100 {
            let st = self.s.state();
            if st.mode() == 5 && st.link.status == 0 && st.full_location() != ow(room) {
                break;
            }
            self.act("WAIT 4");
        }
        (self.s.state().full_location(), self.s.state().link.position.x)
    }
}

/// routes.yaml as landed: `d3first` names its macros, every one off, and no other route has any; the fresh
/// generation's entry turns them on; a room that is no whirlwind step's is refused.
#[test]
fn the_macros_are_a_route_switch_off_as_landed() {
    let landed = GameMap::builtin();
    assert_eq!(landed.macros[ROUTE], Macros::default());
    assert!(!landed.macros.contains_key("d2first"));
    let on = map_on();
    assert_eq!(on.macros[ROUTE].whirlwinds, [ow(0x0b), ow(0x22)].into_iter().collect());
    assert!(on.macros[ROUTE].unaided_docks);
    for room in ["0x45", "0x55"] {
        let bad = ROUTES_YAML.replacen(OFF, &ON.replacen("0x0b", room, 1), 1);
        let e = GameMap::from_yaml(GAME_YAML, &bad).unwrap_err().to_string();
        assert!(e.contains("is no whirlwind step's room"), "{room}: {e}");
    }
}

/// Off (as landed) the networks play the whirlwind by hand: out of Level 6 Link comes out facing down and RECORDER
/// waits for him to face up; on, it is offered at once (the macro turns him) and is no dock warp's.
#[test]
fn the_switch_decides_who_plays_the_whirlwind() {
    let Some(off) = Walk::start(&Arc::new(GameMap::builtin()), "d3first_0_22c_0", 0x22) else { return };
    assert_eq!(off.gc.warp(off.s.state()), None);
    assert!(!off.allowed("RECORDER"), "facing down out of Level 6: the network turns first");
    let Some(on) = Walk::start(&map_on(), "d3first_0_22c_0", 0x22) else { return };
    assert_eq!(on.gc.warp(on.s.state()), Some(Warp::Whirlwind { level: 2 }));
    assert!(on.allowed("RECORDER"), "the macro turns Link itself");
}

/// The macro's RECORDER is masked by Link's position only: wherever he walks in 0x0b and 0x22, with the monsters
/// live, the warp's part of the mask is the same as with them removed.
#[test]
fn the_whirlwind_macro_is_masked_by_position_only() {
    let map = map_on();
    for (state, room) in [("d3first_0_0bc_0", 0x0b), ("d3first_0_22c_1", 0x22), ("d3first_0_0bc_2", 0x0b)] {
        let Some(mut w) = Walk::start(&map, state, room) else { return };
        let mut offered = 0;
        for step in ["WAIT 4", "MOVE E 8", "MOVE E 8", "MOVE S 8", "WAIT 8", "MOVE W 8", "MOVE W 8", "MOVE W 8",
                     "MOVE N 8", "WAIT 8", "MOVE E 8", "MOVE S 8"]
        {
            let (live, none) = w.with_and_without_monsters();
            assert_eq!(live, none, "{state} at {:?}", (w.s.state().link.position.x, w.s.state().link.position.y));
            offered += usize::from(w.allowed("RECORDER"));
            if w.s.state().full_location() != ow(room) || w.s.state().game_over() || !w.allowed(step) {
                break;
            }
            w.act(step);
        }
        assert!(offered > 0, "{state}: the macro is offered somewhere");
    }
}

/// One RECORDER plays the whole whirlwind (the turn, every play the index needs, the wait): out of Level 5 into
/// 0x45 (Level 4's whirlwind, two plays facing down) and out of Level 6 into 0x3c (Level 2's, a turn up and two).
#[test]
fn one_recorder_plays_the_whole_whirlwind() {
    let map = map_on();
    for (state, room, landing) in [("d3first_0_0bc_0", 0x0b, 0x45), ("d3first_0_22c_0", 0x22, 0x3c)] {
        let Some(mut w) = Walk::start(&map, state, room) else { return };
        w.tidy();
        w.wire();
        assert!(w.allowed("RECORDER"), "{state}");
        w.tidy();
        w.act("RECORDER");
        assert_eq!(w.landing(room), (ow(landing), 0x80), "{state}: set down by one RECORDER");
    }
}

/// The dock warp unaided: no give-way (a monster below the dock, a shot in the air), the plays beside the family's
/// own actions, and the warp's part of the mask the same with and without them; as landed (aided) it is not.
#[test]
fn the_unaided_dock_never_looks_at_monsters() {
    let raft_end = |w: &mut Walk| {
        for _ in 0..100 {
            let st = w.s.state();
            let (_, row) = st.room().dock().expect("a dock room");
            if st.mode() == 5 && st.link.status == 0 && st.link.position.y > row && st.link.position.y < row + 8 {
                return;
            }
            w.tidy();
            w.act("WAIT 4");
        }
        panic!("no end of the ride down");
    };
    let Some(mut w) = Walk::start(&map_on(), "d3first_0_55n_v2_0", 0x55) else { return };
    raft_end(&mut w);
    w.tidy();
    w.wire();
    assert!(matches!(w.gc.warp(w.s.state()), Some(Warp::Dock { level: 4, unaided: true })));
    // On the dock: MOVE S and the family's own actions; never a step that boards the raft back up.
    let clear = w.warp_part();
    assert!(w.allowed("MOVE S 8") && w.allowed("WAIT 4") && !w.allowed("MOVE W 8") && !w.allowed("MOVE N 8"));
    w.place(0x07, 0x7E, 0x8D);
    assert_eq!(w.warp_part(), clear, "a red Octorok below the dock changes nothing");
    assert!(w.allowed("MOVE S 8"));
    w.tidy();
    w.wire();
    w.act("MOVE S 8");
    for _ in 0..8 {
        if w.allowed("CLIP N") {
            break;
        }
        w.tidy();
        w.act("MOVE W 8");
    }
    assert!(w.allowed("CLIP N") && w.allowed("WAIT 4"), "the clip beside the family's actions");
    w.tidy();
    w.act("CLIP N");
    let clear = w.warp_part();
    assert!(w.allowed("RECORDER") && w.allowed("WAIT 4"), "on the water spot: the plays and a wait");
    w.place(0x55, 0xC0, 0x7D);
    assert_eq!(w.warp_part(), clear, "a shot in the air changes nothing");
    w.tidy();
    w.act("RECORDER");
    assert_eq!(w.landing(0x55).0, ow(0x34), "the warp lands as aided");

    // As landed the dock gives way: a shot in the air takes the plays away.
    let Some(mut aided) = Walk::start(&Arc::new(GameMap::builtin()), "d3first_0_55n_v2_0", 0x55) else { return };
    raft_end(&mut aided);
    aided.tidy();
    aided.wire();
    aided.act("MOVE S 8");
    for _ in 0..8 {
        if aided.allowed("CLIP N") {
            break;
        }
        aided.tidy();
        aided.act("MOVE W 8");
    }
    aided.tidy();
    aided.act("CLIP N");
    let clear = aided.warp_part();
    aided.place(0x55, 0xC0, 0x7D);
    assert_ne!(aided.warp_part(), clear, "aided, a shot holds the plays back");
}
