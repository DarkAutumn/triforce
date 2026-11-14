//! The wrap exit (round 22; `zelda::clips::SkipKind::WrapExit`) through the route objective and the end conditions:
//! with a step `0x75: entry E, exit W`, `CLIP E` out of 0x75's solid W edge into 0x74 is the correct exit (as the
//! one-room env, the critic and the chain runner judge it: `exit_is_correct`), the path search behind PBRS and the
//! `target` features leads to the E edge on the wrap exit's rows, and with clips off the W exit is unreachable.
//!
//! No state starts in 0x75: Link walks in from `0_38w_0` with `RoomId` set to 0x76 (SETUP, as `zelda-map.py` reaches
//! rooms without states); monsters are removed (SETUP).

mod common;

use common::{ZCore, open_core};
use env::end_conditions::{EXIT_CORRECT, EndCondition, StepView};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use zelda::game_map::{GAME_YAML, ROUTES_YAML};
use zelda::objectives::{GameCompletion, exit_is_correct, wraps_into_wrong_exits};
use zelda::room::Target;
use zelda::{Action, Direction, GameMap, LoadOptions, MapLocation, Session, Var};

/// Two test routes through 0x75 entered from E: the clip way (out W into 0x74) and one that leaves N.
const TEST_ROUTES: &str = "
wrapexit:
- {level: 0, room: '0x76', entry: E, exit: W}
- {level: 0, room: '0x75', entry: E, exit: W}
- {level: 0, room: '0x74', entry: E, exit: cave}
north:
- {level: 0, room: '0x76', entry: E, exit: W}
- {level: 0, room: '0x75', entry: E, exit: N}
";

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn map() -> Arc<GameMap> {
    Arc::new(GameMap::from_yaml(GAME_YAML, &format!("{ROUTES_YAML}{TEST_ROUTES}")).expect("test routes load"))
}

/// 0x75 entered from 0x76 at (`$F0`, `$8D`), monsters removed, Link walked W to rest at `x` on row `y`.
fn in_0x75(map: &Arc<GameMap>, x: i32, y: i32) -> Option<Session<ZCore>> {
    let mut s = Session::new(open_core()?, map.clone());
    s.set_clips(true);
    let state = common::load_named_state("0_38w_0").expect("0_38w_0");
    s.load(&state, &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    s.nes_mut().poke(0xEB, 0x76);
    s.apply_ram(&[]);
    s.act(Action::parse("MOVE W 8").unwrap()).unwrap();
    assert_eq!(s.state().full_location(), ow(0x75));
    for slot in 1..=11u16 {
        s.nes_mut().poke(0x34F + slot, 0);
    }
    s.apply_ram(&[]);
    let at = |s: &Session<ZCore>| {
        let st = s.state();
        (st.link.position.x, st.link.position.y, st.var(Var::LinkGridOffset))
    };
    let walk = |s: &mut Session<ZCore>, buttons: u8, done: &dyn Fn(&Session<ZCore>) -> bool| {
        for _ in 0..300 {
            if done(s) {
                return;
            }
            s.press(buttons, 1);
        }
        panic!("never got there");
    };
    use zelda::executor::buttons::{DOWN, LEFT, UP};
    walk(&mut s, LEFT, &|s| at(s).0 == 0xD0 && at(s).2 == 0);
    walk(&mut s, if y < 0x8D { UP } else { DOWN }, &|s| at(s).1 == y && at(s).2 == 0);
    walk(&mut s, zelda::executor::buttons::RIGHT, &|s| at(s).0 == x && at(s).2 == 0);
    Some(s)
}

/// The step `0x75: entry E, exit W` makes 0x74 the next room; the objective's W target is reached from the E part
/// only with clips, through the wrap exit's jumps (CLIP E from the zone on rows Y `$65`-`$AD`, never from a softlock
/// row); CLIP E then ends in 0x74 and the end condition calls it the correct exit.
#[test]
fn the_wrap_exit_is_the_correct_exit_of_its_step() {
    let map = map();
    let Some(mut s) = in_0x75(&map, 0xE0, 0x8D) else { return };
    let mut gc = GameCompletion::new(map.clone(), "wrapexit", Some(1)).unwrap();
    gc.set_clips(true);
    let objective = gc.objectives(None, s.state());
    assert_eq!(objective.next_rooms, BTreeSet::from([ow(0x74)]));
    assert!(objective.clips);
    assert!(wraps_into_wrong_exits(&map, s.state(), &objective).is_empty(), "the wrap exit is wanted");
    let targets: Vec<Target> = objective.targets.iter().copied().collect();
    let edge: Vec<Target> = (5..=14).map(|r| Target::Tile((0, r))).collect();
    assert_eq!(targets, edge, "the W edge on the wrap exit's rows: {objective:?}");
    let room = s.state().room();
    let wavefront = room.wavefront(&targets, &[], &[], true);
    let here = s.state().link.tile();
    assert!(wavefront.get(here).is_some(), "the zone leads out W");
    let jump = wavefront.jumps_from(here).next().expect("a jump from the zone");
    assert_eq!((jump.direction, jump.to), (Direction::E, (-1, here.1)));
    for jump in room.skip_jumps() {
        assert!((5..=14).contains(&jump.from.1), "a jump on a softlock row: {jump:?}");
    }
    // A softlock row's zone: reachable, but by walking to a safe row first (no jump starts there).
    let softlock = zelda::Position::new(0xE0, 0x5D).tile();
    assert!(wavefront.get(softlock).is_some() && wavefront.jumps_from(softlock).next().is_none());
    // Clips off: the W exit can't be reached from anywhere in the room (no PBRS, no `target` direction).
    gc.set_clips_allowed(false);
    let off = gc.objectives(None, s.state());
    assert!(!off.clips);
    let targets: Vec<Target> = off.targets.iter().copied().collect();
    assert_eq!(room.wavefront(&targets, &[], &[], off.clips).get(here), None);
    // CLIP E: the correct exit.
    let change = s.act(Action::parse("CLIP E").unwrap()).unwrap();
    assert!(change.clip.is_some_and(|c| c.completed()));
    assert_eq!(s.state().full_location(), ow(0x74));
    assert!(exit_is_correct(&map, &change.previous, &objective, ow(0x74)));
    let mut end = EndCondition::RoomExit { rooms: HashSet::from([ow(0x75)]), enter_rooms: HashSet::new(), one_room: true };
    let view = StepView { change: &change, prev_objective: &objective, map: &map, linger_tail_steps: 0 };
    assert_eq!(end.check(&view).as_deref(), Some(EXIT_CORRECT));
    // The arrival is 0x74 entered from E: the next step's entry (a route step `0x74: entry E` follows it).
    assert_eq!(zelda::objectives::route_entry(ow(0x75), ow(0x74)), Some("E"));
}

/// A route step leaving 0x75 any other way vetoes the wrap exit (it would be a wrong exit): no CLIP E in the mask.
#[test]
fn a_step_leaving_another_way_vetoes_the_wrap_exit() {
    let map = map();
    let Some(mut s) = in_0x75(&map, 0xE0, 0x8D) else { return };
    let mut gc = GameCompletion::new(map.clone(), "north", Some(1)).unwrap();
    gc.set_clips(true);
    let objective = gc.objectives(None, s.state());
    assert_eq!(objective.next_rooms, BTreeSet::from([ow(0x65)]));
    let vetoed = wraps_into_wrong_exits(&map, s.state(), &objective);
    assert_eq!(vetoed, [Direction::E]);
    let clip_e = Action::parse("CLIP E").unwrap();
    assert!(s.mask().is_allowed(&clip_e));
    s.set_vetoed_wraps(&vetoed);
    assert!(!s.mask().is_allowed(&clip_e));
}
