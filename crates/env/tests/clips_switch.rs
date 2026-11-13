//! The run-time clips switch through `SpecialistEnv` (round 22): flipped mid-room it leaves the same state as having
//! been off (or on) from the start: the mask, the `clip_usable_*` bits, the objective and the path search behind the
//! `target` features, on the overworld wrap (0x77), in 1:0x22 before and after the block clip, in 4:0x50 before and
//! after its block clip, and on 4:0x21's island before and after the ladder clip. With it off a room with two ways is
//! played the plain way (1:0x22's push, 4:0x50's walk). The clip counters (a clip carried on into the next room, a
//! cut clip finished by a MOVE) and the wrap veto (0x5b's lower E opening).

mod common;

use common::{TfzStore, ZCore, open_core};
use env::config::{GameData, RamRandomization};
use env::observation::{Observation, feature_names};
use env::specialist_env::{EnvOptions, SpecialistEnv, StepResult};
use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};
use zelda::objectives::{Objective, ObjectiveKind, wraps_into_wrong_exits};
use zelda::room::Target;
use zelda::{Action, ActionKind, ActionMask, Direction, MapLocation, Var};

type Env = SpecialistEnv<ZCore, TfzStore>;

static DATA: LazyLock<Arc<GameData>> = LazyLock::new(common::game_data);

/// A room of `family` from `state`, RAM randomization off, CLIP runs even where masked (`lenient`: a run with the
/// switch off replays a CLIP of one with it on), the switch fixed to `clips`, reset with seed 0 and `setup` applied
/// (test setup pokes).
fn room_env(nes: ZCore, family: &str, state: &str, clips: bool, setup: &[(u16, u8)]) -> Env {
    let mut config = (*DATA.specialists[family]).clone();
    config.ram_randomization = RamRandomization::default();
    assert!(config.clips, "{family} has clips: true");
    let opts = EnvOptions { start_state: Some(state.into()), lenient_actions: true, ..Default::default() };
    let mut env = SpecialistEnv::new(Arc::new(config), DATA.clone(), nes, TfzStore::new(), opts).expect("env");
    env.set_clips_allowed(Some(clips));
    env.reset(Some(0), None).unwrap();
    for &(addr, value) in setup {
        zelda::Nes::poke(env.session_mut().nes_mut(), addr, value); // test setup
    }
    env.session_mut().apply_ram(&[]);
    env
}

fn act(env: &mut Env, text: &str) -> StepResult {
    env.step(Action::parse(text).unwrap()).unwrap()
}

/// `clip_usable_{N,S,W,E}` of an observation.
fn usable(obs: &Observation) -> [f32; 4] {
    let names = feature_names();
    let at = names.link.iter().position(|n| n == "clip_usable_N").expect("clip_usable_N");
    obs.link[at..at + 4].try_into().unwrap()
}

/// What a decision is built from: the observation, the mask and the objective.
#[derive(Debug, PartialEq)]
struct Decision {
    obs: Observation,
    mask: ActionMask,
    objective: Objective,
}

fn decision(env: &Env, obs: Observation) -> Decision {
    Decision { obs, mask: *env.session().mask(), objective: env.objective().unwrap().objective.clone() }
}

/// One scenario: a family's room from a state (with setup pokes), the actions before the flip and after it.
struct Scenario {
    family: &'static str,
    state: &'static str,
    setup: Vec<(u16, u8)>,
    before: Vec<&'static str>,
    after: Vec<&'static str>,
}

/// Plays `before` with the switch `from`, flips it to `to`, plays `after`; the same actions with the switch `to`
/// from the start give the same decisions from the first one after the flip on.
fn flip_matches_fixed(nes: ZCore, s: &Scenario, from: bool, to: bool) -> ZCore {
    let label = format!("{} {}: {from} -> {to} after {:?}", s.family, s.state, s.before);
    let play = |nes: ZCore, start: bool, flip: bool| {
        let mut env = room_env(nes, s.family, s.state, start, &s.setup);
        for text in &s.before {
            act(&mut env, text);
        }
        if flip {
            env.set_clips_allowed(Some(to));
        }
        let decisions: Vec<Decision> = s
            .after
            .iter()
            .map(|text| {
                let r = act(&mut env, text);
                assert!(r.info.terminal.is_none(), "{label}: {text} ended the episode");
                decision(&env, r.obs)
            })
            .collect();
        assert_eq!(env.clips_allowed(), to, "{label}");
        (decisions, env.into_parts().0.into_nes())
    };
    let (flipped, nes) = play(nes, from, true);
    let (fixed, nes) = play(nes, to, false);
    assert_eq!(flipped, fixed, "{label}");
    // The switch decides the mask's CLIP and the observation's bits together.
    for d in &flipped {
        let clip = d.mask.kind[ActionKind::Clip.index()];
        assert_eq!(usable(&d.obs).iter().any(|&b| b > 0.0), clip, "{label}");
        assert!(to || !clip, "{label}: CLIP offered with the switch off");
        assert_eq!(d.objective.clips, to, "{label}: the objective's path search");
    }
    nes
}

/// From the E door of 1:$22 into the corner below the east block (G = ($A0, $9D)).
const BELOW_EAST_BLOCK: [&str; 6] = ["MOVE W 32", "MOVE W 32", "MOVE W 8", "MOVE S 8", "MOVE W 8", "MOVE W 8"];

fn scenarios() -> Vec<Scenario> {
    // 4:$50: Link at ($B0, $8D), on the grid, under the block round 21's experiment clips N into.
    let at_50 = vec![(Var::DoorwayDir.addr(), 0), (0x70, 0xB0), (0x84, 0x8D), (0x394, 0)];
    // 4:$21: the Gels gone (slots 1-11 emptied), so nothing interrupts.
    let gels_gone: Vec<(u16, u8)> = (1..12).map(|slot| ((zelda::tables::OBJ_ID.0 + slot) as u16, 0)).collect();
    let mut s = vec![
        // The overworld start room 0x77: in the W wrap zone (X <= $20 on the start row).
        Scenario {
            family: "ow-start",
            state: "start",
            setup: vec![],
            before: vec!["MOVE W 32", "MOVE W 32", "MOVE W 16", "MOVE W 4"],
            after: vec!["WAIT 4", "MOVE N 4", "WAIT 4"],
        },
        Scenario {
            family: "trap-room",
            state: "1_22e_0",
            setup: vec![],
            before: BELOW_EAST_BLOCK.to_vec(),
            after: vec!["WAIT 4", "MOVE E 8"],
        },
        Scenario {
            family: "trap-room",
            state: "1_22e_0",
            setup: vec![],
            before: [&BELOW_EAST_BLOCK[..], &["CLIP N"]].concat(),
            after: vec!["WAIT 4", "MOVE W 8"],
        },
        Scenario {
            family: "vire",
            state: "4_50e_0",
            setup: at_50.clone(),
            before: vec!["WAIT 4"],
            after: vec!["WAIT 4", "MOVE W 8"],
        },
        Scenario {
            family: "vire",
            state: "4_50e_0",
            setup: at_50,
            before: vec!["CLIP N"],
            after: vec!["WAIT 4", "MOVE W 8"],
        },
        Scenario {
            family: "stalfos-keese-gel",
            state: "4_21s_0",
            setup: gels_gone.clone(),
            before: vec!["MOVE N 4", "MOVE N 32", "MOVE N 32"],
            after: vec!["WAIT 4", "MOVE S 8"],
        },
    ];
    s.push(Scenario {
        family: "stalfos-keese-gel",
        state: "4_21s_0",
        setup: gels_gone,
        before: vec!["MOVE N 4", "MOVE N 32", "MOVE N 32", "CLIP N"],
        after: vec!["WAIT 4", "MOVE S 8"],
    });
    s
}

/// Flipped off mid-room (in the wrap zone, below 1:$22's block and 8 pixels into it after the clip, under 4:$50's
/// block and inside it after the clip, at 4:$21's water and beyond it after the ladder clip) the decisions are the
/// ones the switch off from the start gives; flipped back on, the ones of the switch on from the start.
#[test]
fn a_flip_mid_room_equals_the_switch_fixed_from_the_start() {
    let Some(mut nes) = open_core() else { return };
    for s in scenarios() {
        nes = flip_matches_fixed(nes, &s, true, false);
        nes = flip_matches_fixed(nes, &s, false, true);
    }
}

/// The objective of the state the actions reach, with the switch `clips`, the last observation and the room's
/// stairs tiles.
fn objective_after(nes: ZCore, s: &Scenario, clips: bool) -> (Objective, Observation, BTreeSet<Target>, ZCore) {
    let mut env = room_env(nes, s.family, s.state, clips, &s.setup);
    let mut obs = None;
    for text in s.before.iter().chain(&s.after) {
        obs = Some(act(&mut env, text).obs);
    }
    let objective = env.objective().unwrap().objective.clone();
    let stairs = env.session().state().room().stairs_tiles().into_iter().map(Target::Tile).collect();
    (objective, obs.unwrap(), stairs, env.into_parts().0.into_nes())
}

/// The family's `clips: false`, as before round 21, in the state the actions reach.
fn objective_without_clips(nes: ZCore, s: &Scenario) -> (Objective, ZCore) {
    let mut config = (*DATA.specialists[s.family]).clone();
    config.ram_randomization = RamRandomization::default();
    config.clips = false;
    let opts = EnvOptions { start_state: Some(s.state.into()), lenient_actions: true, ..Default::default() };
    let mut env = SpecialistEnv::new(Arc::new(config), DATA.clone(), nes, TfzStore::new(), opts).expect("env");
    env.reset(Some(0), None).unwrap();
    for &(addr, value) in &s.setup {
        zelda::Nes::poke(env.session_mut().nes_mut(), addr, value);
    }
    env.session_mut().apply_ram(&[]);
    for text in s.before.iter().chain(&s.after) {
        act(&mut env, text);
    }
    let objective = env.objective().unwrap().objective.clone();
    (objective, env.into_parts().0.into_nes())
}

/// The rooms with a clip way and a plain way inside one room need no route change: with the switch off the objective
/// is the plain one (the family's `clips: false`, as before round 21). In 1:$22 below the east block that is the
/// push: the targets are the tiles Link pushes the west block from, not the stairs the clip reaches with the switch on.
/// In 4:$50 the exit is the same either way, but the path search behind the `target` features only jumps with it on.
#[test]
fn with_the_switch_off_a_room_is_played_the_plain_way() {
    let Some(mut nes) = open_core() else { return };
    let before = BELOW_EAST_BLOCK.to_vec();
    let room_22 = Scenario { family: "trap-room", state: "1_22e_0", setup: vec![], before, after: vec![] };
    let at_50 = vec![(Var::DoorwayDir.addr(), 0), (0x70, 0xB0), (0x84, 0x8D), (0x394, 0)];
    let room_50 = Scenario { family: "vire", state: "4_50e_0", setup: at_50, before: vec!["WAIT 4"], after: vec![] };
    for s in [&room_22, &room_50] {
        let (off, off_obs, stairs, n) = objective_after(nes, s, false);
        let (plain, n) = objective_without_clips(n, s);
        let (on, on_obs, _, n) = objective_after(n, s, true);
        nes = n;
        let plain_way = (&plain.targets, &plain.next_rooms, plain.kind);
        assert_eq!((&off.targets, &off.next_rooms, off.kind), plain_way, "{}", s.state);
        assert_eq!(off.next_rooms, on.next_rooms, "{}: the same exit", s.state);
        if s.state == "1_22e_0" {
            assert_ne!(off.targets, on.targets, "with the switch on the clip reaches the stairs without the push");
            assert!(!stairs.is_empty() && on.targets == stairs, "on: the stairs, {:?}", on.targets);
            assert!(off.targets.is_disjoint(&stairs), "off: the push spots, {:?}", off.targets);
        } else {
            assert_ne!(off_obs.target, on_obs.target, "the jump shortens the way only with the switch on");
        }
    }
}

/// 4:$21 with the N wall already open (bombed from below the clip's end, then back over the water by `CLIP S`): a
/// `CLIP N` from the island's water runs on through the hole into $11 in one action, and counts as done.
#[test]
fn a_clip_carried_on_into_the_next_room_counts_as_done() {
    let Some(nes) = open_core() else { return };
    let gels_gone: Vec<(u16, u8)> = (1..12).map(|slot| ((zelda::tables::OBJ_ID.0 + slot) as u16, 0)).collect();
    let mut env = room_env(nes, "stalfos-keese-gel", "4_21s_0", true, &gels_gone);
    let plan = ["MOVE N 4", "MOVE N 32", "MOVE N 32", "CLIP N", "MOVE S 8", "BOMB N", "WAIT 32", "WAIT 32", "WAIT 32"];
    for text in plan {
        act(&mut env, text);
    }
    assert!(env.session().mask().is_allowed(&Action::parse("MOVE N 16").unwrap()), "the wall is open");
    // Back over the water onto the island, up to the water's edge.
    let mut r = act(&mut env, "CLIP S");
    assert!(r.info.terminal.is_none(), "back on the island");
    for _ in 0..4 {
        if env.session().mask().is_allowed(&Action::parse("CLIP N").unwrap()) {
            break;
        }
        r = act(&mut env, "MOVE N 8");
    }
    assert!(r.info.terminal.is_none());
    assert!(env.session().mask().is_allowed(&Action::parse("CLIP N").unwrap()), "CLIP N at the water");
    let r = act(&mut env, "CLIP N");
    assert_eq!(env.session().state().full_location(), MapLocation::new(4, 0x11, false), "carried on into $11");
    let t = r.info.terminal.as_ref().expect("the room change ends the episode");
    assert_eq!(t.ending, "exit-correct");
    // Done: the first ladder clip, CLIP S, and this one; nothing aborted.
    assert_eq!((t.clip_stats.done, t.clip_stats.done_n, t.clip_stats.done_s, t.clip_stats.aborted), (3, 2, 1, 0));
}

/// A block clip cut short leaves Link part way into the square: a `MOVE` along the clip's line that ends at least as
/// far past the skipped point as the clip would have counts as finishing it (`finished_by_move`), once; a MOVE that
/// stops short of that, or runs on another line, doesn't. Checked on 1:$22's east block with the rule's states.
#[test]
fn a_move_that_crosses_a_cut_clips_square_finishes_it() {
    let Some(nes) = open_core() else { return };
    let mut env = room_env(nes, "trap-room", "1_22e_0", true, &[]);
    for text in BELOW_EAST_BLOCK {
        act(&mut env, text);
    }
    let start = env.session().state().clone();
    let target = zelda::clips::skip_target(&start, Direction::N).expect("the block clip N below the east block");
    let outcome = zelda::clips::ClipOutcome {
        direction: Direction::N,
        kind: Some(target.kind),
        abort: Some(zelda::clips::ClipAbort::Stalled),
        presses: 1,
        target: Some(target),
    };
    // Cut 6 pixels into the square (T is $95: Y $8F), on the clip's line X $A0.
    let at = |env: &mut Env, x: u8, y: u8| {
        for (addr, value) in [(0x70, x), (0x84, y), (0x394, 0)] {
            zelda::Nes::poke(env.session_mut().nes_mut(), addr, value); // test setup
        }
        env.session_mut().apply_ram(&[]).clone()
    };
    let cut = at(&mut env, 0xA0, 0x8F);
    let open = zelda::clips::OpenClip::of(&outcome, &start, &cut).expect("an aborted clip in its room");
    let move_n = Action::parse("MOVE N 8").unwrap();
    let through = at(&mut env, 0xA0, 0x8D);
    let short = at(&mut env, 0xA0, 0x8E);
    let aside = at(&mut env, 0x98, 0x8D);
    assert!(open.finished_by(&move_n, &cut, &through), "8 pixels into the square: as far as the clip");
    assert!(!open.finished_by(&move_n, &cut, &short), "short of the clip's end");
    assert!(!open.finished_by(&move_n, &cut, &aside), "another line");
    assert!(!open.finished_by(&Action::parse("MOVE W 8").unwrap(), &cut, &through), "another heading");
    let completed = zelda::clips::ClipOutcome { abort: None, ..outcome };
    assert!(zelda::clips::OpenClip::of(&completed, &start, &through).is_none(), "a completed clip leaves nothing open");
}

/// 0x5b entered from W (objective: E into 0x5c through the upper E opening, game.yaml's `span`): on rows Y `$B5` and
/// `$BD` the W wrap would land at the lower E opening, a wrong exit, so it is not offered (mask and `clip_usable_W`);
/// on the rows whose landing is inside the span it is. An objective that leaves S keeps the wrap: walking S from the
/// landing is another use of it. Every row line of the W zone is tried (X $10).
#[test]
fn a_wrap_that_lands_only_on_a_wrong_exit_is_not_offered() {
    let Some(nes) = open_core() else { return };
    let place = |y: u8| vec![(0x70, 0x10), (0x84, y), (0x394, 0), (Var::LinkDirection.addr(), 2)];
    let mut nes = Some(nes);
    let (mut vetoed_rows, mut offered_rows) = (Vec::new(), Vec::new());
    for y in (0x3Du8..=0xDD).step_by(8) {
        let mut env = room_env(nes.take().unwrap(), "ow-moblin", "0_5bw_0", true, &place(y));
        let r = act(&mut env, "WAIT 4");
        let state = env.session().state().clone();
        let at_row = state.link.position.x == 0x10 && state.link.position.y == i32::from(y);
        if at_row && zelda::clips::skip_available(&state, Direction::W) {
            let objective = env.objective().unwrap().objective.clone();
            assert!(objective.next_rooms.contains(&MapLocation::new(0, 0x5c, false)), "{objective:?}");
            let vetoed = !wraps_into_wrong_exits(&DATA.map, &state, &objective).is_empty();
            let offered = env.session().mask().is_allowed(&Action::parse("CLIP W").unwrap());
            assert_eq!(offered, !vetoed, "Y ${y:02X}");
            assert_eq!(usable(&r.obs)[2], if offered { 1.0 } else { 0.0 }, "Y ${y:02X}");
            if vetoed {
                let south = Objective {
                    kind: ObjectiveKind::Move,
                    next_rooms: [MapLocation::new(0, 0x6b, false)].into_iter().collect(),
                    ..objective
                };
                assert!(wraps_into_wrong_exits(&DATA.map, &state, &south).is_empty(), "S keeps the wrap");
                vetoed_rows.push(y);
            } else {
                offered_rows.push(y);
            }
        }
        nes = Some(env.into_parts().0.into_nes());
    }
    assert_eq!(vetoed_rows, [0xB5, 0xBD], "offered on {offered_rows:02X?}");
    assert!(!offered_rows.is_empty(), "a row inside the span keeps the wrap");
}

/// The cut clip in the real game: 4:$21's ladder clip N with the Gels alive is often hit on the ladder (round 21:
/// most of its aborted clips), leaving Link over the water; a `MOVE N 16` from there carries him onto the land beyond
/// it, which finishes the clip (`Change::clip_finished_by_move`, counted once in `ClipStats::finished_by_move`; the
/// clip stays in `aborted`). The first seed of 64 whose clip is cut by a hit.
#[test]
fn a_ladder_clip_cut_by_a_hit_and_finished_by_a_move_is_counted() {
    let Some(mut nes) = open_core() else { return };
    let (clip_n, move_n) = (Action::parse("CLIP N").unwrap(), Action::parse("MOVE N 16").unwrap());
    for seed in 0..64 {
        let mut env = room_env(nes, "stalfos-keese-gel", "4_21s_0", true, &[]);
        env.reset(Some(seed), None).unwrap();
        let s = env.session_mut();
        for _ in 0..20 {
            if zelda::clips::skip_available(s.state(), Direction::N) {
                break;
            }
            s.act(Action::parse("MOVE N 8").unwrap()).unwrap();
        }
        let mut stats = zelda::clips::ClipStats::default();
        let cut = s.act(clip_n).unwrap();
        stats.update(&cut);
        let damage = zelda::clips::ClipAbort::Interrupted(zelda::Interrupt::Damage);
        let hit = cut.clip.is_some_and(|c| c.abort == Some(damage));
        if hit && s.state().full_location() == MapLocation::new(4, 0x21, false) && s.state().link.position.y > 0x5D {
            let on = s.act(move_n).unwrap();
            stats.update(&on);
            assert!(on.clip_finished_by_move, "seed {seed}: Y ${:02X}", s.state().link.position.y);
            let again = s.act(Action::parse("MOVE N 8").unwrap()).unwrap();
            stats.update(&again);
            assert!(!again.clip_finished_by_move, "counted once");
            assert_eq!((stats.done, stats.aborted, stats.finished_by_move), (0, 1, 1), "seed {seed}");
            return;
        }
        nes = env.into_parts().0.into_nes();
    }
    panic!("no ladder clip cut by a hit in 64 seeds");
}
