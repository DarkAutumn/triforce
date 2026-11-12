//! The grid-point skip through `SpecialistEnv` (round 21): the `clips:` config block, the mask per design, the
//! `clip_usable_*` observation, PBRS shaping (`jump`, `walk`), the counters and the episode's target in the terminal
//! info. Room `$77` from the game start (objective: the sword cave; W and E are wrong exits) for the wrap; level 1
//! room `$22` from its E door for the block clip.

mod common;

use common::{TfzStore, ZCore, open_core};
use env::config::{ClipAction, ClipShaping, ClipsConfig, GameData, RamRandomization, SpecialistConfig};
use env::critic::walk_cap;
use env::observation::{Observation, feature_names};
use env::specialist_env::{EnvOptions, SpecialistEnv, StepResult};
use std::sync::{Arc, LazyLock};
use zelda::{Action, ActionKind, Direction};

type Env = SpecialistEnv<ZCore, TfzStore>;

static DATA: LazyLock<Arc<GameData>> = LazyLock::new(common::game_data);

fn make_env(nes: ZCore, clips: ClipsConfig) -> Env {
    let mut config = (*DATA.specialists["ow-start"]).clone();
    config.ram_randomization = RamRandomization::default();
    config.clips = clips;
    let opts = EnvOptions { start_state: Some("start".into()), ..Default::default() };
    SpecialistEnv::new(Arc::new(config), DATA.clone(), nes, TfzStore::new(), opts).expect("env")
}

fn clips(action: ClipAction, shaping: ClipShaping) -> ClipsConfig {
    ClipsConfig { action, shaping }
}

fn act(env: &mut Env, text: &str) -> StepResult {
    env.step(Action::parse(text).unwrap()).unwrap()
}

fn link_x(env: &Env) -> i32 {
    env.session().state().link.position.x
}

/// `clip_usable_{N,S,W,E}` of an observation.
fn usable(obs: &Observation) -> [f32; 4] {
    let names = feature_names();
    let at = names.link.iter().position(|n| n == "clip_usable_N").expect("clip_usable_N");
    obs.link[at..at + 4].try_into().unwrap()
}

/// From the start ($78, $8D) MOVE W until Link is in the W wrap zone (X <= $20).
fn walk_into_the_zone(env: &mut Env) -> StepResult {
    env.reset(Some(0), None).unwrap();
    let mut r = act(env, "MOVE W 32");
    while link_x(env) > 0x20 {
        r = act(env, "MOVE W 4");
    }
    assert_eq!(env.session().state().location(), 0x77);
    r
}

fn pbrs(r: &StepResult) -> f64 {
    r.info
        .rewards
        .iter()
        .filter(|(name, _)| name.contains("pbrs"))
        .map(|(_, v)| v)
        .sum()
}

#[test]
fn every_specialist_defaults_to_no_clips_and_the_block_parses() {
    for (name, config) in &DATA.specialists {
        assert_eq!(config.clips, ClipsConfig::default(), "{name}");
    }
    assert_eq!(ClipsConfig::default(), clips(ClipAction::None, ClipShaping::Jump));
    let base = std::fs::read_to_string(common::repo().join("specialists/ow-start.yaml")).unwrap();
    for (block, want) in [
        ("{action: kind}", Some(clips(ClipAction::Kind, ClipShaping::Jump))),
        ("{action: long, shaping: walk}", Some(clips(ClipAction::Long, ClipShaping::Walk))),
        ("{action: edge, shaping: jump}", Some(clips(ClipAction::Edge, ClipShaping::Jump))),
        ("{action: sword}", None),
        ("{action: kind, wrap: true}", None),
        ("{shaping: run}", None),
    ] {
        let text = format!("{base}\nclips: {block}\n");
        let parsed = SpecialistConfig::from_yaml(&text, "test", Some(&DATA.map)).ok().map(|c| c.clips);
        assert_eq!(parsed, want, "{block}");
    }
}

/// The mask per design in the zone, and `clip_usable_W` set for every design but `none`.
#[test]
fn each_design_gates_the_mask_and_the_observation() {
    let Some(mut nes) = open_core() else { return };
    let clip_w = Action::parse("CLIP W").unwrap();
    for action in [ClipAction::None, ClipAction::Kind, ClipAction::Long, ClipAction::Edge] {
        let mut env = make_env(nes, clips(action, ClipShaping::Jump));
        let (obs, _) = env.reset(Some(0), None).unwrap();
        assert_eq!(usable(&obs), [0.0; 4], "{action:?} at the start");
        let r = walk_into_the_zone(&mut env);
        let want = if action == ClipAction::None { 0.0 } else { 1.0 };
        assert_eq!(usable(&r.obs), [0.0, 0.0, want, 0.0], "{action:?} in the zone");
        assert_eq!(env.session().mask().is_allowed(&clip_w), action == ClipAction::Kind, "{action:?}");
        assert_eq!(env.session().mask().kind[ActionKind::Clip.index()], action == ClipAction::Kind);
        nes = env.into_parts().0.into_nes();
    }
}

/// `kind`: CLIP W wraps to X $F0 and is counted; MOVE E out of $77 then is a wrong exit, whose terminal info carries
/// the counters and the episode's route target and entry.
#[test]
fn a_clip_is_counted_and_the_episode_carries_its_target() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, clips(ClipAction::Kind, ClipShaping::Jump));
    walk_into_the_zone(&mut env);
    let r = act(&mut env, "CLIP W");
    assert_eq!(link_x(&env), 0xF0);
    assert!(r.info.terminal.is_none());
    let r = act(&mut env, "MOVE E 8");
    let t = r.info.terminal.as_ref().expect("the wrong exit ends the episode");
    assert_eq!((t.clip_stats.done, t.clip_stats.done_w, t.clip_stats.aborted), (1, 1, 0));
    assert_eq!((t.target.as_str(), t.entry.as_str()), ("cave", "start"));
    let json = r.info.to_json();
    assert_eq!(json["clips"]["done"], 1);
    assert_eq!(json["target"], "cave");
}

/// `jump` pays the whole wavefront difference of the wrap; `walk` caps it at what walking the step's frames could
/// earn (frames × 1.5 / 8 / 20), either sign.
#[test]
fn walk_shaping_caps_the_wraps_pbrs() {
    let Some(nes) = open_core() else { return };
    let mut paid = Vec::new();
    let mut nes = Some(nes);
    for shaping in [ClipShaping::Jump, ClipShaping::Walk] {
        let mut env = make_env(nes.take().unwrap(), clips(ClipAction::Kind, shaping));
        walk_into_the_zone(&mut env);
        let r = act(&mut env, "CLIP W");
        assert_eq!(link_x(&env), 0xF0);
        paid.push((pbrs(&r), r.info.frames));
        nes = Some(env.into_parts().0.into_nes());
    }
    let ((jump, frames), (walk, walk_frames)) = (paid[0], paid[1]);
    assert_eq!(frames, walk_frames, "the same run");
    let cap = walk_cap(frames);
    assert!(jump.abs() > cap, "the wrap jumps farther than a walk: {jump} vs cap {cap}");
    assert!((walk - jump.clamp(-cap, cap)).abs() < 1e-9, "walk {walk} jump {jump} cap {cap}");
}

/// `edge` in the env: the objective (the cave) makes W a wrong exit, so MOVE W 8 in the zone wraps.
#[test]
fn the_edge_design_wraps_out_of_a_wrong_exit() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, clips(ClipAction::Edge, ClipShaping::Jump));
    walk_into_the_zone(&mut env);
    let settings = env.session().clip_settings();
    assert_eq!(settings.correct_exits & (Direction::W.value() | Direction::E.value()), 0);
    act(&mut env, "MOVE W 8");
    assert_eq!((link_x(&env), env.session().state().location()), (0xF0, 0x77));
}

/// The trap-room family's config in level 1 room $22 from its E door (`1_22e_0`; the route's target is the stairs,
/// exit `cave`, with the push spots as targets while the block is pending), RAM randomization off.
fn make_room_22_env(nes: ZCore, clips: ClipsConfig) -> Env {
    let mut config = (*DATA.specialists["trap-room"]).clone();
    config.ram_randomization = RamRandomization::default();
    config.clips = clips;
    let opts = EnvOptions { start_state: Some("1_22e_0".into()), ..Default::default() };
    SpecialistEnv::new(Arc::new(config), DATA.clone(), nes, TfzStore::new(), opts).expect("env")
}

/// From the E door of $22 by `moves`; Link's (X, Y) and the last observation's `clip_usable_*`.
fn walk_room_22(env: &mut Env, moves: &[&str]) -> ((i32, i32), [f32; 4]) {
    env.reset(Some(0), None).unwrap();
    let mut obs = None;
    for text in moves {
        obs = Some(act(env, text).obs);
    }
    let p = env.session().state().link.position;
    ((p.x, p.y), usable(&obs.unwrap()))
}

/// The block clip in a dungeon room per design (level 1 room $22): `clip_usable_S` above the east block,
/// `clip_usable_N` on G below it (for every design but `none`; never W or E); the design's own action (`CLIP N`,
/// `MOVE N 32`, `MOVE N 8`) clips to ($A0, $8D), and MOVE W 32 then takes the stairs: the episode ends a success on
/// the route's exit (`cave`, entered from E) with the clip counted.
#[test]
fn the_block_clip_in_a_dungeon_room_per_design() {
    let Some(mut nes) = open_core() else { return };
    let to_corner = ["MOVE W 32", "MOVE W 32", "MOVE W 8"];
    for (action, clip) in [
        (ClipAction::None, None),
        (ClipAction::Kind, Some("CLIP N")),
        (ClipAction::Long, Some("MOVE N 32")),
        (ClipAction::Edge, Some("MOVE N 8")),
    ] {
        let mut env = make_room_22_env(nes, clips(action, ClipShaping::Jump));
        let want = if action == ClipAction::None { 0.0 } else { 1.0 };
        let above = [&to_corner[..], &["MOVE N 16", "MOVE W 8", "MOVE W 8"]].concat();
        assert_eq!(walk_room_22(&mut env, &above), ((0xA0, 0x75), [0.0, want, 0.0, 0.0]), "{action:?} above");
        let below = [&to_corner[..], &["MOVE S 8", "MOVE W 8", "MOVE W 8"]].concat();
        assert_eq!(walk_room_22(&mut env, &below), ((0xA0, 0x9D), [want, 0.0, 0.0, 0.0]), "{action:?} below");
        assert_eq!(env.session().mask().is_allowed(&Action::parse("CLIP N").unwrap()), action == ClipAction::Kind);
        if let Some(clip) = clip {
            let r = act(&mut env, clip);
            let p = env.session().state().link.position;
            assert_eq!((p.x, p.y, r.info.terminal.is_none()), (0xA0, 0x8D, true), "{action:?} {clip}");
            let r = act(&mut env, "MOVE W 32");
            let t = r.info.terminal.as_ref().expect("the stairs end the episode");
            assert_eq!((t.ending.as_str(), t.outcome), ("exit-correct", "success"), "{action:?}");
            assert_eq!((t.clip_stats.done, t.clip_stats.done_n, t.clip_stats.aborted), (1, 1, 0), "{action:?}");
            assert_eq!((t.target.as_str(), t.entry.as_str()), ("cave", "E"), "{action:?}");
        }
        nes = env.into_parts().0.into_nes();
    }
}
