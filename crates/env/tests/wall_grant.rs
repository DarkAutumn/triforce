//! Round 40, the fresh generation (the owner, Nov 23, backlog "A cheated wall bomb appears only when the wall is the
//! objective"; routes.yaml `bomb_rules: {wall_grant: objective}` on `d3first`): a room's cheated wall bomb is in
//! Link's bag exactly while the objective's target is the bomb wall the route leaves through. Played in chains from
//! `d3first` states with an empty bag, at every decision: one bomb while `env::objective::wall_is_next`, none
//! otherwise; and the same rooms under `wall_grant: room` (2.0.0.x's rule) hand the bomb over on entering. The
//! weapon rooms (Dodongo's 2:0x0e, Manhandla's 3:0x4d) keep theirs whatever the objective: `boss_bombs.rs` plays
//! both on `d3first`. The take-back is the ledger's unit test `the_wall_grant_follows_the_objective`. Skipped
//! without the ROM.
//!
//! SETUP (RAM writes): the bag emptied on the start state; in 3:0x5b's last pass the Darknuts removed and the room's
//! treasure marked taken (the fight and the treasure done), Link's hearts refilled.

mod common;

use common::{TfzStore, ZCore, open_core};
use env::StateStore;
use env::chain::{ChainEnv, ChainStatus, Route};
use env::config::GameData;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::cheats::CheatItem;
use zelda::game_map::GameMap;
use zelda::objectives::ObjectiveKind;
use zelda::{Action, MapLocation, Nes};

type C = ChainEnv<ZCore, TfzStore>;

const BOMBS: u16 = 0x658;
const OBJ_TYPE: u16 = 0x34F;
const FLAG_ITEM_TAKEN: u8 = 0x10;

/// The repo's data, or the same with `d3first`'s wall grant back on the room rule.
fn data(room_rule: bool) -> Arc<GameData> {
    let data = common::game_data();
    if !room_rule {
        return data;
    }
    let root = common::repo();
    let read = |rel: &str| std::fs::read_to_string(root.join(rel)).unwrap();
    let routes = read("triforce/routes.yaml");
    let on = "    wall_grant: objective\n";
    assert!(routes.contains(on));
    let mut data = (*data).clone();
    data.map = Arc::new(GameMap::from_yaml(&read("triforce/game.yaml"), &routes.replacen(on, "", 1)).unwrap());
    Arc::new(data)
}

/// A chain on `d3first` from `state` with the bag emptied.
fn empty_bag(nes: ZCore, data: Arc<GameData>, dir: &std::path::Path, state: &str) -> C {
    let mut source = common::read_state_file(&common::state_path(state)).unwrap();
    source.machine.poke(BOMBS, 0);
    let path = dir.join(format!("{state}.state"));
    TfzStore::new().save(&path, &source).unwrap();
    let route = Route::load(&data.map, &data.states_dir, "d3first", None, Some(path.to_str().unwrap())).unwrap();
    let mut c = ChainEnv::new(data, nes, TfzStore::new(), route);
    let ChainStatus::Act { .. } = c.reset(Some(0), &mut ChaCha8Rng::seed_from_u64(0)).unwrap() else {
        panic!("{state}: the run ended at reset")
    };
    c
}

/// The rule at the current decision: one bomb, granted, exactly while the wall is the objective's target.
fn check(c: &C, what: &str) -> bool {
    let state = c.session().state();
    let objective = c.objective().unwrap();
    let next = env::objective::wall_is_next(objective, state, c.acting_reward().unwrap());
    assert_eq!(state.bombs(), i32::from(next), "{what}: a bomb while the wall is next ({:?})", objective.kind);
    assert_eq!(c.context().1.room.bombs, u8::from(next), "{what}: that bomb is the room's grant");
    next
}

/// WAITs in place, the rule checked at every decision; whether the wall was ever next.
fn wait_and_check(c: &mut C, steps: usize, what: &str) -> bool {
    let mut seen = check(c, what);
    for _ in 0..steps {
        let wait = Action::parse("WAIT 4").unwrap();
        if !matches!(c.step(wait).unwrap(), ChainStatus::Act { .. }) {
            break;
        }
        let tidy = c.session().state().health() < c.session().state().max_health() as f64;
        if tidy {
            let hc = c.session().state().hearts_and_containers();
            c.session_mut().apply_ram(&[(zelda::Var::HeartsAndContainers, (hc & 0xF0) | (hc >> 4))]);
        }
        seen |= check(c, what);
    }
    seen
}

/// A wall room the route enters for something else first: no bomb while the objective is the cellar's stairs (7:0x1a
/// with the red candle still to take) or the N exit (3:0x5b's first pass, crossed for one Darknut's bombs); the room
/// rule gives one on entering.
#[test]
fn no_wall_bomb_while_the_objective_is_elsewhere() {
    let Some(mut nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    for (state, room) in [("d3first_7_1aw_0", MapLocation::new(7, 0x1a, false)), ("d3first_3_5bs_0", MapLocation::new(3, 0x5b, false))] {
        let data = data(false);
        assert!(zelda::cheats::bomb_opens(&data.map, room), "{state}: its bomb opens a wall");
        let mut c = empty_bag(nes, data, dir.path(), state);
        assert_eq!(c.session().state().full_location(), room);
        assert!(!wait_and_check(&mut c, 30, state), "{state}: the wall was never the objective");
        assert_eq!(c.cheated()[CheatItem::Bombs], 0);
        nes = c.into_parts().0.into_nes();

        let mut old = empty_bag(nes, data_room(), dir.path(), state);
        assert_eq!(old.session().state().bombs(), 1, "{state}: the room rule hands the bomb over on entering");
        assert_eq!(old.context().1.room.bombs, 1);
        old.step(Action::parse("WAIT 4").unwrap()).unwrap();
        nes = old.into_parts().0.into_nes();
    }
}

fn data_room() -> Arc<GameData> {
    data(true)
}

/// The wall is the objective at once (7:0x1a out of the candle's cellar, E through its bomb wall): the bomb is there
/// from the first decision.
#[test]
fn the_wall_bomb_is_there_when_the_wall_is_next() {
    let Some(nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    let mut c = empty_bag(nes, data(false), dir.path(), "d3first_7_1ac_0");
    assert!(check(&c, "7:0x1a from the cellar"), "the E wall is the objective on entering");
    wait_and_check(&mut c, 10, "7:0x1a from the cellar");
}

/// 3:0x5b's last pass (W -> E) with an empty bag: no bomb during the fight for the room's bombs; with the fight and
/// the treasure done (SETUP) the E wall is next and the bomb comes.
#[test]
fn the_last_pass_of_0x5b_gets_its_bomb_after_the_fight() {
    let Some(nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    let mut c = empty_bag(nes, data(false), dir.path(), "d3first_3_5bw_0");
    for _ in 0..10 {
        c.step(Action::parse("WAIT 4").unwrap()).unwrap();
        if c.objective().unwrap().kind == ObjectiveKind::Fight {
            break;
        }
    }
    assert_eq!(c.objective().unwrap().kind, ObjectiveKind::Fight, "the last pass fights for the room's bombs");
    assert!(!wait_and_check(&mut c, 5, "0x5b's fight"), "no bomb during the fight");
    let state = c.session().state().clone();
    let flags = zelda::room_state::room_flags_addr(state.ram(), 0x5b).expect("the level's room flags");
    let s = c.session_mut();
    for slot in 1..=11u16 {
        if state.ram()[(OBJ_TYPE + slot) as usize] == zelda::enums::enemy::RED_DARKNUT {
            Nes::poke(s.nes_mut(), OBJ_TYPE + slot, 0);
        }
    }
    Nes::poke(s.nes_mut(), flags as u16, state.ram()[flags] | FLAG_ITEM_TAKEN);
    s.apply_ram(&[]);
    assert!(wait_and_check(&mut c, 10, "0x5b, the fight and treasure done"), "the E wall became the objective");
}
