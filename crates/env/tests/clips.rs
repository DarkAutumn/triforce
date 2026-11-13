//! The grid-point skip through `SpecialistEnv` (round 21): the `clips:` switch, the mask, the `clip_usable_*`
//! observation, PBRS over a skip, the counters and the episode's target in the terminal info. Room `$77` from the
//! game start (objective: the sword cave; W and E are wrong exits) for the wrap; level 1 room `$22` from its E door
//! for the block clip.

mod common;

use common::{TfzStore, ZCore, open_core};
use env::config::{GameData, RamRandomization, SpecialistConfig};
use env::observation::{Observation, feature_names};
use env::specialist_env::{EnvOptions, SpecialistEnv, StepResult};
use std::sync::{Arc, LazyLock};
use zelda::{Action, ActionKind};

type Env = SpecialistEnv<ZCore, TfzStore>;

static DATA: LazyLock<Arc<GameData>> = LazyLock::new(common::game_data);

fn make_env(nes: ZCore, clips: bool) -> Env {
    let mut config = (*DATA.specialists["ow-start"]).clone();
    config.ram_randomization = RamRandomization::default();
    config.clips = clips;
    let opts = EnvOptions { start_state: Some("start".into()), ..Default::default() };
    SpecialistEnv::new(Arc::new(config), DATA.clone(), nes, TfzStore::new(), opts).expect("env")
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

/// `clips:` is one switch, `true` or `false` (default false); the removed designs' block (`{action: ..., shaping:
/// ...}`) is refused.
#[test]
fn the_clips_switch_parses() {
    // ow-start says `clips: true`; without that line the switch is off.
    let text = std::fs::read_to_string(common::repo().join("specialists/ow-start.yaml")).unwrap();
    let base: String = text.lines().filter(|l| !l.starts_with("clips:")).map(|l| format!("{l}\n")).collect();
    for (value, want) in [
        ("true", Some(true)),
        ("false", Some(false)),
        ("{action: kind}", None),
        ("{action: none, shaping: jump}", None),
        ("long", None),
    ] {
        let text = format!("{base}clips: {value}\n");
        let parsed = SpecialistConfig::from_yaml(&text, "test", Some(&DATA.map)).ok().map(|c| c.clips);
        assert_eq!(parsed, want, "{value}");
    }
    let parsed = SpecialistConfig::from_yaml(&base, "test", Some(&DATA.map)).unwrap();
    assert!(!parsed.clips, "default off");
}

/// In the zone the mask has CLIP W and `clip_usable_W` is set with clips on; neither with clips off.
#[test]
fn the_switch_gates_the_mask_and_the_observation() {
    let Some(mut nes) = open_core() else { return };
    let clip_w = Action::parse("CLIP W").unwrap();
    for clips in [false, true] {
        let mut env = make_env(nes, clips);
        let (obs, _) = env.reset(Some(0), None).unwrap();
        assert_eq!(usable(&obs), [0.0; 4], "clips {clips} at the start");
        let r = walk_into_the_zone(&mut env);
        let want = if clips { 1.0 } else { 0.0 };
        assert_eq!(usable(&r.obs), [0.0, 0.0, want, 0.0], "clips {clips} in the zone");
        assert_eq!(env.session().mask().is_allowed(&clip_w), clips, "clips {clips}");
        assert_eq!(env.session().mask().kind[ActionKind::Clip.index()], clips);
        nes = env.into_parts().0.into_nes();
    }
}

/// CLIP W wraps to X $F0 and is counted; PBRS pays the wavefront difference (the wrap is no jump of the path search:
/// the step pays what the new tile's distance says); MOVE E out of $77 then is a wrong exit, whose terminal info
/// carries the counters and the episode's route target and entry.
#[test]
fn a_clip_is_counted_and_the_episode_carries_its_target() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, true);
    walk_into_the_zone(&mut env);
    let r = act(&mut env, "CLIP W");
    assert_eq!(link_x(&env), 0xF0);
    assert!(r.info.terminal.is_none());
    assert!(pbrs(&r) != 0.0, "the wrap moves Link across the room's wavefront");
    let r = act(&mut env, "MOVE E 8");
    let t = r.info.terminal.as_ref().expect("the wrong exit ends the episode");
    assert_eq!((t.clip_stats.done, t.clip_stats.done_w, t.clip_stats.aborted), (1, 1, 0));
    assert_eq!((t.target.as_str(), t.entry.as_str()), ("cave", "start"));
    let json = r.info.to_json();
    assert_eq!(json["clips"]["done"], 1);
    assert_eq!(json["target"], "cave");
}

/// The trap-room family's config in level 1 room $22 from its E door (`1_22e_0`; the route's target is the stairs,
/// exit `cave`), RAM randomization off.
fn make_room_22_env(nes: ZCore, clips: bool) -> Env {
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

/// The block clip in a dungeon room (level 1 room $22): `clip_usable_S` above the east block, `clip_usable_N` on G
/// below it (with clips on; never W or E); CLIP N clips to ($A0, $8D), and MOVE W 32 then takes the stairs: the
/// episode ends a success on the route's exit (`cave`, entered from E) with the clip counted. With clips on the
/// objective needs no push (the search reaches the stairs through the clip), so the clip pays PBRS.
#[test]
fn the_block_clip_in_a_dungeon_room() {
    let Some(mut nes) = open_core() else { return };
    let to_corner = ["MOVE W 32", "MOVE W 32", "MOVE W 8"];
    for clips in [false, true] {
        let mut env = make_room_22_env(nes, clips);
        let want = if clips { 1.0 } else { 0.0 };
        let above = [&to_corner[..], &["MOVE N 16", "MOVE W 8", "MOVE W 8"]].concat();
        assert_eq!(walk_room_22(&mut env, &above), ((0xA0, 0x75), [0.0, want, 0.0, 0.0]), "clips {clips} above");
        let below = [&to_corner[..], &["MOVE S 8", "MOVE W 8", "MOVE W 8"]].concat();
        assert_eq!(walk_room_22(&mut env, &below), ((0xA0, 0x9D), [want, 0.0, 0.0, 0.0]), "clips {clips} below");
        assert_eq!(env.session().mask().is_allowed(&Action::parse("CLIP N").unwrap()), clips);
        if clips {
            let r = act(&mut env, "CLIP N");
            let p = env.session().state().link.position;
            assert_eq!((p.x, p.y, r.info.terminal.is_none()), (0xA0, 0x8D, true), "CLIP N");
            assert!(pbrs(&r) > 0.0, "the clip leads to the stairs: {:?}", r.info.rewards);
            let r = act(&mut env, "MOVE W 32");
            let t = r.info.terminal.as_ref().expect("the stairs end the episode");
            assert_eq!((t.ending.as_str(), t.outcome), ("exit-correct", "success"));
            assert_eq!((t.clip_stats.done, t.clip_stats.done_n, t.clip_stats.aborted), (1, 1, 0));
            assert_eq!((t.target.as_str(), t.entry.as_str()), ("cave", "E"));
        }
        nes = env.into_parts().0.into_nes();
    }
}

/// The route's step 4:0x21 entered from S (round 21: through 0x31's N door onto the island) with the
/// stalfos-keese-gel family (`clips: true`), RAM randomization off, the Gels emptied after the reset.
fn make_room_21_env(nes: ZCore) -> Env {
    let mut config = (*DATA.specialists["stalfos-keese-gel"]).clone();
    config.ram_randomization = RamRandomization::default();
    assert!(config.clips, "the family leaves the island by the ladder clip");
    let opts = EnvOptions { start_state: Some("4_21s_0".into()), ..Default::default() };
    let mut env = SpecialistEnv::new(Arc::new(config), DATA.clone(), nes, TfzStore::new(), opts).expect("env");
    env.reset(Some(0), None).unwrap();
    for slot in 1..12 {
        zelda::Nes::poke(env.session_mut().nes_mut(), (zelda::tables::OBJ_ID.0 + slot) as u16, 0);
    }
    env.session_mut().apply_ram(&[]);
    env
}

/// Link's Y.
fn link_y(env: &Env) -> i32 {
    env.session().state().link.position.y
}

/// The route step 4:0x21 S -> N by actions: `MOVE N` onto the island (Link stops on T, Y $7D, at the water), `CLIP N`
/// (the ladder clip, from T: back off to G' and over the two squares of water to Y $55, 8 pixels into the top wall),
/// `MOVE S 8` off the wall, `BOMB N`, `WAIT` for the blast, `MOVE N` through the hole into the rupee room 0x11: the
/// episode ends `exit-correct`. BOMB N from the clip's end (Y $55) spends the bomb and leaves the wall shut: the bomb
/// lands outside the room, so the step down is needed (the objective's bomb spot is ($78, $5D)).
#[test]
fn the_route_step_through_4_21_by_the_ladder_clip() {
    let Some(nes) = open_core() else { return };
    let mut env = make_room_21_env(nes);
    let mut frames = 0;
    for text in ["MOVE N 4", "MOVE N 32", "MOVE N 32"] {
        frames += act(&mut env, text).info.frames;
    }
    assert_eq!((link_x(&env), link_y(&env)), (0x78, 0x7D), "stopped on T at the water");
    assert!(env.session().mask().is_allowed(&Action::parse("CLIP N").unwrap()));
    let r = act(&mut env, "CLIP N");
    frames += r.info.frames;
    assert_eq!((link_x(&env), link_y(&env), r.info.terminal.is_none()), (0x78, 0x55, true), "over the water");
    let move_n = Action::parse("MOVE N 16").unwrap();
    // From the clip's end the bomb is spent for nothing.
    let end = env.save();
    let bombs = env.session().state().bombs();
    assert!(bombs > 0 && env.session().mask().is_allowed(&Action::parse("BOMB N").unwrap()), "{bombs} bombs");
    act(&mut env, "BOMB N");
    for _ in 0..4 {
        act(&mut env, "WAIT 32");
    }
    // The cheat ledger gives the room's bomb back (0x21 needs one), but the wall is still shut.
    assert_eq!(env.session().state().full_location(), zelda::MapLocation::new(4, 0x21, false));
    assert!(!env.session().mask().is_allowed(&move_n), "the wall is still shut");
    env.load(&end);
    // One step down, then the bomb opens the wall.
    for text in ["MOVE S 8", "BOMB N"] {
        frames += act(&mut env, text).info.frames;
    }
    // The wall opens at the end of the blast (about 80 frames after the press).
    for _ in 0..3 {
        frames += act(&mut env, "WAIT 32").info.frames;
    }
    let mut r = act(&mut env, "MOVE N 16");
    frames += r.info.frames;
    for _ in 0..3 {
        if r.info.terminal.is_some() {
            break;
        }
        r = act(&mut env, "MOVE N 16");
        frames += r.info.frames;
    }
    let t = r.info.terminal.as_ref().expect("the bombed wall ends the episode");
    assert_eq!((t.ending.as_str(), t.outcome), ("exit-correct", "success"));
    assert_eq!((t.target.as_str(), t.entry.as_str()), ("N", "S"));
    assert_eq!((t.clip_stats.done, t.clip_stats.done_n, t.clip_stats.aborted), (1, 1, 0));
    assert_eq!(env.session().state().full_location(), zelda::MapLocation::new(4, 0x11, false));
    println!("4:0x21 S -> N: {frames} frames, {} steps", t.episode_steps);
    assert!(frames < 450, "{frames} frames");
}

/// A route exit's direction letter as the route writes it.
fn exit_of(env: &Env) -> String {
    let (route, step) = env.route();
    DATA.map.routes[route][step.expect("a route step")].exit.clone()
}

/// The experiment route `clips-gauntlet` (round 21) after Dungeon 4 joined `d2first` (round 20): a specialist naming
/// it draws both of its targets for every start state of the gauntlet (from W: E or S, from E: W or S), and it
/// leaves every registered family's resets as they are without it: the start states of ow-zora and ow-moblin (whose
/// rooms 0x59-0x5b it shares) and of every family with a Dungeon 4 room reset to the same `d2first` step with and
/// without the route in routes.yaml.
#[test]
fn the_clips_gauntlet_draws_its_targets_and_leaves_the_families_alone() {
    let (Some(nes), Some(nes2)) = (open_core(), open_core()) else { return };
    let root = common::repo();
    let text = std::fs::read_to_string(root.join("experiments/r21/gauntlet-kind-jump.yaml")).unwrap();
    let config = SpecialistConfig::from_yaml(&text, "gauntlet-kind-jump", Some(&DATA.map)).unwrap();
    let starts = config.resolve_start_states(&root.join("experiments/r21/states"));
    assert_eq!(starts.len(), 18);
    let mut env = SpecialistEnv::new(Arc::new(config), DATA.clone(), nes, TfzStore::new(), EnvOptions::default())
        .expect("gauntlet env");
    for path in &starts {
        let path = path.to_str().unwrap();
        let entry = if path.contains("e_") { "E" } else { "W" };
        let mut drawn = std::collections::BTreeSet::new();
        for seed in 0..40 {
            env.reset(Some(seed), Some(path)).unwrap();
            assert_eq!(env.route().0, "clips-gauntlet", "{path}");
            drawn.insert(exit_of(&env));
        }
        let want: std::collections::BTreeSet<String> =
            [if entry == "W" { "E" } else { "W" }, "S"].into_iter().map(str::to_owned).collect();
        assert_eq!(drawn, want, "{path}");
    }
    let nes = env.into_parts().0.into_nes();

    // The same game data without the route.
    let routes = std::fs::read_to_string(root.join("triforce/routes.yaml")).unwrap();
    let (head, tail) = routes.split_once("# clips-gauntlet").expect("the route");
    let tail = &tail[tail.find("\n# Equal-length alternates").expect("after it")..];
    let game = std::fs::read_to_string(root.join("triforce/game.yaml")).unwrap();
    let map = zelda::GameMap::from_yaml(&game, &format!("{head}{}", &tail[1..])).unwrap();
    assert!(!map.routes.contains_key("clips-gauntlet") && map.routes.len() + 1 == DATA.map.routes.len());
    let without = Arc::new(GameData { map: Arc::new(map), ..(**DATA).clone() });
    let families = [
        "ow-zora", "ow-moblin", "ow-octorok", "ow-tektite", "ow-peahat", "vire", "like-like", "boss-gleeok", "zol",
        "stalfos-keese-gel", "equipment-room", "trap-room",
    ];
    let (mut nes, mut nes2) = (Some(nes), Some(nes2));
    let mut resets = 0;
    for name in families {
        let config = DATA.specialists[name].clone();
        let starts = config.resolve_start_states(&DATA.states_dir);
        let mut with = SpecialistEnv::new(config.clone(), DATA.clone(), nes.take().unwrap(), TfzStore::new(), EnvOptions::default()).unwrap();
        let mut alone = SpecialistEnv::new(config, without.clone(), nes2.take().unwrap(), TfzStore::new(), EnvOptions::default()).unwrap();
        for path in &starts {
            let path = path.to_str().unwrap();
            for seed in 0..3 {
                with.reset(Some(seed), Some(path)).unwrap();
                alone.reset(Some(seed), Some(path)).unwrap();
                let (a, b) = (with.route(), alone.route());
                assert_eq!((a.0, a.1), (b.0, b.1), "{name} {path} seed {seed}");
                assert_eq!(a.0, "d2first", "{name} {path} seed {seed}");
                resets += 1;
            }
        }
        nes = Some(with.into_parts().0.into_nes());
        nes2 = Some(alone.into_parts().0.into_nes());
    }
    assert!(resets > 300, "{resets} resets");
}
