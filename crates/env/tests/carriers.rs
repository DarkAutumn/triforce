//! Round 40, the fresh generation (decisions 2026-11-23 "The fresh generation's Dungeon 3"; routes.yaml `bomb_rules:
//! {carrier_targets: true}`): 3:0x5b's first pass on `d3first` played in a chain from `d3first_3_5bs_0` with the drop
//! cycle (`$52A` WorldKillCycle) and the bag written. At cycle 0 every Red Darknut is a bomb carrier (its death reads
//! column 1 of drop group 2: bombs, asm:Drop_SetUpItem) and the objective aims at the three as well as at the N exit;
//! the observation's `bomb_carrier` column (`entity_threat`'s last, behind `obs.threat`) names the same three. At
//! cycle 1 (a kill was made on the way, or the first Darknut died without a drop) and with a full bag, neither does:
//! the room is a plain crossing. The goals the visit's record carries say the same. Skipped without the ROM.
//!
//! SETUP (RAM writes): the cycle and the bag on the start state, before the run loads it.

mod common;

use common::{TfzStore, ZCore, open_core};
use env::StateStore;
use env::chain::{ChainEnv, ChainStatus, Route};
use env::observation::{ENTITY_THREAT_SIZE, Observation};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use zelda::enums::{Direction, enemy};
use zelda::objectives::{ObjectiveKind, Role};
use zelda::{Action, MapLocation};

type C = ChainEnv<ZCore, TfzStore>;

/// `$52A` WorldKillCycle, `$658` the bag, `$67C` its size.
const CYCLE: u16 = 0x52A;
const BOMBS: u16 = 0x658;
const BOMB_MAX: u16 = 0x67C;
const ROOM: MapLocation = MapLocation::new(3, 0x5b, false);

/// The first pass's start state with `cycle` and `bombs` (`None`: the bag full) written, saved under its own name
/// (the name says the route step) in `dir`, played on `d3first` with `darknut` observing the threat block; the
/// step's `carriers: fight` (round 43) set to `fight` whatever routes.yaml ships.
fn first_pass(nes: ZCore, dir: &Path, cycle: u8, bombs: Option<u8>, fight: bool) -> (C, Box<Observation>) {
    let mut data = (*common::game_data()).clone();
    let mut map = (*data.map).clone();
    let step = map.routes.get_mut("d3first").unwrap().iter_mut().find(|s| s.location() == ROOM && s.exit == "N");
    step.expect("0x5b's first pass").carrier_fight = fight;
    data.map = Arc::new(map);
    let data = Arc::new(data);
    let mut source = common::read_state_file(&common::state_path("d3first_3_5bs_0")).unwrap();
    source.machine.poke(CYCLE, cycle);
    let max = source.machine.peek(BOMB_MAX);
    source.machine.poke(BOMBS, bombs.unwrap_or(max));
    let path = dir.join("d3first_3_5bs_0.state");
    TfzStore::new().save(&path, &source).unwrap();
    let route = Route::load(&data.map, &data.states_dir, "d3first", None, Some(path.to_str().unwrap())).unwrap();
    let mut c = ChainEnv::new(data.clone(), nes, TfzStore::new(), route);
    let mut config = (*data.specialist("darknut").unwrap()).clone();
    config.obs.threat = true;
    c.configs.insert("darknut".into(), Arc::new(config));
    let ChainStatus::Act { obs, .. } = c.reset(Some(0), &mut ChaCha8Rng::seed_from_u64(0)).unwrap() else {
        panic!("the run ended at reset")
    };
    assert_eq!(c.session().state().full_location(), ROOM);
    assert_eq!(c.session().state().world_kill_cycle(), cycle, "nothing on the way writes the cycle");
    (c, obs)
}

/// WAIT until the room's three Red Darknuts are out (their spawn takes a few frames of play).
fn darknuts_out(c: &mut C, mut obs: Box<Observation>) -> Box<Observation> {
    let wait = Action::parse("WAIT 4").unwrap();
    for _ in 0..40 {
        let st = c.session().state();
        if st.enemies.iter().filter(|e| e.id == enemy::RED_DARKNUT && e.is_active()).count() == 3 {
            return obs;
        }
        match c.step(wait).unwrap() {
            ChainStatus::Act { obs: next, .. } => obs = next,
            ChainStatus::Done(r) => panic!("the run ended waiting: {}", r.reason),
        }
    }
    panic!("the three Darknuts never came out")
}

/// The Darknuts' (slot, type).
fn darknuts(c: &C) -> BTreeSet<(u8, u8)> {
    c.session().state().enemies.iter().filter(|e| e.id == enemy::RED_DARKNUT).map(|e| (e.index, e.id)).collect()
}

/// Observation slots whose `bomb_carrier` is lit, as RAM slots.
fn carrier_column(obs: &Observation) -> BTreeSet<u8> {
    (0..obs.entity_threat.len()).filter(|&i| obs.entity_threat[i][ENTITY_THREAT_SIZE - 1] > 0.0).map(|i| i as u8 + 1).collect()
}

/// The objective crosses N: the exit is a target, and the objective is MOVE.
fn assert_crossing(c: &C, what: &str) {
    let objective = c.objective().unwrap();
    assert_eq!(objective.kind, ObjectiveKind::Move, "{what}: the first pass crosses");
    assert!(objective.labels.iter().any(|(_, r)| *r == Role::Exit(Direction::N)), "{what}: the N exit is a target");
    assert!(objective.bag_drops, "{what}: dropped bombs are taken as under a floor");
}

#[test]
fn at_cycle_0_the_darknuts_are_targets_beside_the_exit() {
    let Some(nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (mut c, obs) = first_pass(nes, dir.path(), 0, Some(2), false);
    let obs = darknuts_out(&mut c, obs);
    assert_crossing(&c, "cycle 0");
    let want = darknuts(&c);
    assert_eq!(want.len(), 3);
    assert_eq!(c.objective().unwrap().carriers(), want, "every Darknut is a carrier target");
    let slots: BTreeSet<u8> = want.iter().map(|&(slot, _)| slot).collect();
    assert_eq!(carrier_column(&obs), slots, "the observation names the same three");
    let record = c.result().unwrap().rooms.last().unwrap().goals;
    assert!(record.expected.carriers && record.cycle_in == 0 && record.carrier_seen && !record.fought, "{record:?}");

    // The cycle moves on (the first Darknut died without a drop: 59% at cycle 0, asm:Drop_GroupRates): no carrier
    // is left, the room is a plain crossing.
    c.session_mut().apply_ram(&[(zelda::Var::WorldKillCycle, 1)]);
    let ChainStatus::Act { obs, .. } = c.step(Action::parse("WAIT 4").unwrap()).unwrap() else { panic!("ended") };
    assert_crossing(&c, "cycle 1 after the kill");
    assert!(c.objective().unwrap().carriers().is_empty());
    assert!(carrier_column(&obs).is_empty());
}

#[test]
fn at_cycle_1_the_room_is_a_plain_crossing() {
    let Some(nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (mut c, obs) = first_pass(nes, dir.path(), 1, Some(2), false);
    let obs = darknuts_out(&mut c, obs);
    assert_crossing(&c, "cycle 1");
    assert!(c.objective().unwrap().carriers().is_empty(), "no Darknut's death reads a bomb column at cycle 1");
    assert!(carrier_column(&obs).is_empty());
    let record = c.result().unwrap().rooms.last().unwrap().goals;
    assert!(record.expected.carriers && record.cycle_in == 1 && !record.carrier_seen, "{record:?}");
}

#[test]
fn with_a_full_bag_no_darknut_is_a_carrier() {
    let Some(nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (mut c, obs) = first_pass(nes, dir.path(), 0, None, false);
    let obs = darknuts_out(&mut c, obs);
    assert_eq!(c.session().state().bombs(), c.session().state().bomb_max());
    assert_crossing(&c, "a full bag");
    assert!(c.objective().unwrap().carriers().is_empty(), "the bag has no room");
    assert!(carrier_column(&obs).is_empty());
}

/// Round 43, `carriers: fight` (the owner: "it should force kill that darknut if the part of its observation is lit
/// up"): at cycle 0 the three Darknuts are the only targets, the N exit none (no aim, no PBRS) but still a correct
/// room (`next_rooms`: a chain that leaves goes on); once the cycle moves on (any kill: by the sword, a bomb, with or
/// without a drop) the room is the plain crossing again. At cycle 1 and with a full bag the flag changes nothing.
#[test]
fn with_carriers_fight_the_kill_comes_before_the_door() {
    let Some(nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (mut c, obs) = first_pass(nes, dir.path(), 0, Some(2), true);
    let obs = darknuts_out(&mut c, obs);
    let objective = c.objective().unwrap();
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert_eq!(objective.carriers(), darknuts(&c), "every Darknut is a carrier target");
    assert!(objective.carriers_first(), "the carriers come first");
    assert!(!objective.labels.iter().any(|(_, r)| matches!(r, Role::Exit(_))), "the N exit is no target yet");
    assert!(!objective.pbrs_targets.is_empty() && objective.pbrs_targets.iter().all(|t| objective.targets.contains(t)));
    assert!(objective.next_rooms.contains(&MapLocation::new(3, 0x4b, false)), "0x4b stays the correct room");
    assert_eq!(carrier_column(&obs).len(), 3);

    c.session_mut().apply_ram(&[(zelda::Var::WorldKillCycle, 1)]);
    c.step(Action::parse("WAIT 4").unwrap()).unwrap();
    assert_crossing(&c, "cycle 1 after the kill");
    assert!(!c.objective().unwrap().carriers_first());

    for (cycle, bombs, what) in [(1, Some(2), "cycle 1"), (0, None, "a full bag")] {
        let Some(nes) = open_core() else { return };
        let dir = tempfile::tempdir().unwrap();
        let (mut c, obs) = first_pass(nes, dir.path(), cycle, bombs, true);
        darknuts_out(&mut c, obs);
        assert_crossing(&c, what);
        assert!(!c.objective().unwrap().carriers_first(), "{what}");
    }
}
