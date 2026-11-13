//! Sessions on zcore from the base movie's end: snapshot/restore replays random play exactly.
#![cfg(feature = "zcore")]

mod common;

use common::{open_core, repo};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zcore::ZCore;
use zelda::actions::{NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS};
use zelda::{Action, Direction, GameMap, LoadOptions, Nes, Session};

struct Play<'a>(&'a mut ZCore);

impl movie::Playback for Play<'_> {
    fn run_frame(&mut self, buttons: u8) {
        self.0.run_frame(buttons, false);
    }
    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value);
    }
}

/// Plays `power_on_to_start` on `nes` and returns the machine state at its end.
fn base(nes: &mut ZCore) -> zcore::MachineState {
    let movies = repo().join("corpus/movies");
    let m = movie::Movie::load(&movies.join("power_on_to_start.tfmovie")).unwrap();
    movie::play(&mut Play(nes), &m, &movies).unwrap();
    nes.snapshot()
}

fn random_action(mask: &zelda::ActionMask, rng: &mut ChaCha8Rng) -> Action {
    let kinds: Vec<usize> = (0..NUM_KINDS).filter(|&i| mask.kind[i]).collect();
    let kind = kinds[rng.random_range(0..kinds.len())];
    let dirs: Vec<usize> = (0..NUM_DIRECTIONS).filter(|&i| mask.direction[kind][i]).collect();
    let durs: Vec<usize> = (0..NUM_DURATIONS).filter(|&i| mask.duration[kind][i]).collect();
    let mut pick = |v: &[usize]| if v.is_empty() { 0 } else { v[rng.random_range(0..v.len())] };
    let (d, u) = (pick(&dirs), pick(&durs));
    Action::from_indices(kind, d, u).unwrap()
}

#[test]
fn zcore_snapshot_restore_is_exact() {
    let Some(mut z) = open_core() else { return };
    let b = base(&mut z);
    let mut s = Session::new(z, Arc::new(GameMap::builtin()));
    s.load(&b, &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    let snap = s.snapshot();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let actions: Vec<Action> = (0..60)
        .map(|_| {
            let a = random_action(s.mask(), &mut rng);
            s.act(a).unwrap();
            a
        })
        .collect();
    let (ram, prg) = (*s.nes().ram(), *s.nes().prg_ram());
    s.restore(&snap);
    for a in actions {
        s.act(a).unwrap();
    }
    assert_eq!(&ram, s.nes().ram());
    assert_eq!(&prg, s.nes().prg_ram());
}

/// A sword stab into a Darknut's face plays the shield clink ($604 bit 0) and does no damage; it must not count as
/// a projectile block (blocks need a blockable shot in flight with its bounce signature) nor as a hit. A stab in
/// the back hurts him. (Lee, Nov 9: no false-positive blocks in Darknut rooms.)
#[test]
fn darknut_front_clink_is_not_a_block_or_a_hit() {
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    let machine = common::load_named_state("3_5bs_0");
    let opts = LoadOptions {
        randomize_rng: true,
        fresh_room: true,
        controllable: true,
        overrides: Vec::new(),
    };
    // A Darknut well inside the room, so Link fits on either side of him (the state's layout decides where they
    // stand after 2 s; try seeds until one is).
    let interior = |e: &zelda::game::Enemy| {
        (48..=192).contains(&e.position.x) && (101..=165).contains(&e.position.y) && e.distance > 48.0
    };
    let seed = (0..64u64)
        .find(|&seed| {
            s.load(&machine, &opts, &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
            s.press(0, 120);
            s.state().enemies.iter().any(|e| matches!(e.id, 0x0B | 0x0C) && e.spawn_state == 0 && interior(e))
        })
        .expect("a seed with a Darknut inside the room");
    for (front, expect_damage) in [(true, false), (false, true)] {
        s.load(&machine, &opts, &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
        s.press(0, 120);
        let state = s.state().clone();
        let darknut = state
            .enemies
            .iter()
            .find(|e| matches!(e.id, 0x0B | 0x0C) && e.spawn_state == 0 && interior(e))
            .expect("a Darknut");
        let slot = darknut.index as u16;
        let peek = |s: &mut Session<ZCore>, a: u16| s.nes_mut().peek(a);
        let (x, y, dir) = (peek(&mut s, 0x70 + slot), peek(&mut s, 0x84 + slot), peek(&mut s, 0x98 + slot));
        // Offset to the side his face points at (front) or the opposite side (back); Link faces him.
        let (dx, dy, face): (i16, i16, u8) = match dir {
            1 => (16, 0, 2),
            2 => (-16, 0, 1),
            4 => (0, 16, 8),
            _ => (0, -16, 4),
        };
        let sign = if front { 1 } else { -1 };
        let face = if front { face } else { [0, 2, 1, 0, 8, 0, 0, 0, 4][face as usize] };
        s.nes_mut().poke(0x70, (x as i16 + sign * dx) as u8);
        s.nes_mut().poke(0x84, (y as i16 + sign * dy) as u8);
        s.nes_mut().poke(0x98, face);
        // Hold the Darknut still (stun timer): one that turns toward Link during the swing makes a back stab a
        // front one.
        s.nes_mut().poke(0x3D + slot, 0x40);
        let hp = peek(&mut s, 0x485 + slot);
        let text = match face {
            1 => "SWORD E",
            2 => "SWORD W",
            4 => "SWORD S",
            _ => "SWORD N",
        };
        let change = s.act(Action::parse(text).unwrap()).unwrap();
        assert_eq!(change.blocks(), 0, "front={front}");
        assert_eq!(peek(&mut s, 0x485 + slot) != hp, expect_damage, "front={front}");
        assert_eq!(change.hits() > 0, expect_damage, "front={front}");
    }
}

/// F038/F067: in a boulder room, waiting earns no hits (boulders that leave the screen and Zora dives are no kills),
/// the boulders and their invisible generator are hazards, never enemies; a cave's old man is no projectile.
#[test]
fn boulders_are_hazards_and_waiting_hits_nothing() {
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    let opts = LoadOptions { randomize_rng: true, fresh_room: true, controllable: true, overrides: Vec::new() };
    s.load(&common::load_named_state("0_18e_0"), &opts, &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    let wait = Action::parse("WAIT 32").unwrap();
    let mut boulders = 0;
    for step in 0..300 {
        // Keep Link alive: hearts = containers.
        let hc = s.nes_mut().peek(0x66F);
        s.nes_mut().poke(0x66F, (hc & 0xF0) | (hc >> 4));
        let change = s.act(wait).unwrap();
        assert!(change.enemies_hit.is_empty(), "step {step}: {:?}", change.enemies_hit);
        let state = s.state();
        assert!(state.enemies.iter().all(|e| !matches!(e.id, 0x1F | 0x20)), "step {step}");
        boulders += state.hazards.iter().filter(|h| h.id == 0x20).count();
    }
    assert!(boulders > 0, "boulders fell");

    s.load(&common::load_named_state("0_2ccave_0"), &opts, &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    s.press(0, 120);
    assert!(s.state().in_cave());
    let state = s.state();
    assert!(state.projectiles.is_empty() && state.enemies.is_empty(), "{state:?}");
    // The cave's two standing fires hurt on touch: hazards.
    assert!(!state.hazards.is_empty() && state.hazards.iter().all(|h| h.id == 0x40), "{:?}", state.hazards);
}

/// F070: the Octorok and Darknut names match their ObjHP in the shipped states (red Octoroks 1 HP, blue 2; Red
/// Darknut 4, Blue 8).
#[test]
fn octorok_and_darknut_colours_match_their_hit_points() {
    use zelda::enums::enemy::*;
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    let mut seen = std::collections::BTreeSet::new();
    let mut names: Vec<_> = std::fs::read_dir(repo().join("triforce/states"))
        .unwrap()
        .filter_map(|e| e.unwrap().file_name().into_string().ok()?.strip_suffix(".state").map(str::to_owned))
        .filter(|n| n.starts_with("0_") || n.starts_with("3_"))
        .collect();
    names.sort();
    for name in names {
        s.load(&common::load_named_state(&name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0))
            .unwrap();
        s.press(0, 90);
        for e in s.state().enemies.clone() {
            let expected = match e.id {
                RED_SLOW_OCTOROK | RED_OCTOROK => 1,
                BLUE_SLOW_OCTOROK | BLUE_OCTOROK => 2,
                RED_DARKNUT => 4,
                BLUE_DARKNUT => 8,
                _ => continue,
            };
            if e.is_dying() || e.spawn_state != 0 {
                continue;
            }
            assert_eq!(e.health, expected, "{name}: type ${:02X}", e.id);
            seen.insert(e.id);
        }
    }
    assert!(seen.contains(&RED_SLOW_OCTOROK) && seen.contains(&RED_DARKNUT), "{seen:?}");
}

/// F160: an attack turned perpendicular while Link is between grid points first walks him on to the grid point,
/// as the game would; he never ends up beside the S doorway of 1:0x53 inside the wall band, nor on a solid tile.
#[test]
fn a_perpendicular_attack_off_grid_aligns_link_first() {
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    s.load(&common::load_named_state("1_53s_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(1))
        .unwrap();
    s.nes_mut().poke(0x658, 4); // bombs
    s.nes_mut().poke(0x67C, 8); // max bombs
    for a in ["MOVE N 4", "BOMB W", "MOVE N 4", "MOVE S 4", "MOVE N 8"] {
        s.act(Action::parse(a).unwrap()).unwrap();
        let (x, y) = (s.nes_mut().peek(0x70), s.nes_mut().peek(0x84));
        assert!(y <= 0xBD || x == 0x78, "after {a}: Link at ({x},{y}) beside the doorway");
        assert_eq!(s.nes_mut().peek(0x394) == 0 || s.nes_mut().peek(0x98) & 0x0C != 0, true, "after {a}");
    }
    // Random legal play: whenever Link stands on a grid point, the tile under his feet is walkable.
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for step in 0..2000 {
        let hc = s.nes_mut().peek(0x66F);
        s.nes_mut().poke(0x66F, (hc & 0xF0) | (hc >> 4));
        let a = random_action(s.mask(), &mut rng);
        s.act(a).unwrap();
        let state = s.state().clone();
        if state.full_location() != zelda::MapLocation::new(1, 0x53, false) {
            break;
        }
        // On a grid point inside the room's floor (the doorways' tiles are door codes).
        let (lx, ly) = (s.nes_mut().peek(0x70), s.nes_mut().peek(0x84));
        if s.nes_mut().peek(0x394) == 0 && state.mode() == 5 && (0x20..=0xD0).contains(&lx) && (0x5E..0xBD).contains(&ly) {
            let (x, y) = (s.nes_mut().peek(0x70) as usize, s.nes_mut().peek(0x84) as usize);
            let row = (y + 0x0B - 0x40) >> 3;
            let first_unwalkable = s.nes_mut().peek(0x34A);
            for col in [x >> 3, (x >> 3) + 1] {
                let tile = s.nes_mut().peek((0x6530 + col * 22 + row) as u16);
                assert!(tile < first_unwalkable, "step {step} {a}: Link at ({x},{y}) on tile {tile:#x}");
            }
        }
    }
}

/// F182: over random legal play in the Bubble room (3:0x49) and Manhandla's room (3:0x4d), every legal attack
/// option starts its weapon (the step runs the weapon's animation rather than a 1-3 frame no-op).  One test per
/// room, so the two run in parallel.
#[test]
fn every_legal_attack_starts_its_weapon_in_the_bubble_room() {
    every_legal_attack_starts_its_weapon("3_49s_0");
}

#[test]
fn every_legal_attack_starts_its_weapon_in_manhandlas_room() {
    every_legal_attack_starts_its_weapon("3_4dw_0");
}

fn every_legal_attack_starts_its_weapon(name: &str) {
    use zelda::ActionKind;
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    s.load(&common::load_named_state(name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0))
        .unwrap();
    s.nes_mut().poke(0x658, 8);
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    for step in 0..500 {
        let hc = s.nes_mut().peek(0x66F);
        s.nes_mut().poke(0x66F, (hc & 0xF0) | (hc >> 4));
        if s.state().game_over() || s.state().mode() != 5 {
            break;
        }
        let snap = s.snapshot();
        let mask = s.mask().clone();
        for kind in [ActionKind::Sword, ActionKind::Beams, ActionKind::Boomerang, ActionKind::Arrow, ActionKind::Bomb] {
            for d in mask.directions(kind) {
                let a = Action::new(kind, d, 0).unwrap();
                // An off-grid perpendicular attack first walks Link to the grid (F160); what happens to him on
                // the way is the room's doing, not the mask's.
                let facing_vertical = matches!(d, Direction::N | Direction::S | Direction::NW | Direction::NE | Direction::SW | Direction::SE);
                let line_vertical = s.nes_mut().peek(0x98) & 0x0C != 0;
                if s.nes_mut().peek(0x394) != 0 && facing_vertical != line_vertical {
                    continue;
                }
                let bombs = s.nes_mut().peek(0x658);
                let c = s.act(a).unwrap();
                let started = match kind {
                    ActionKind::Bomb => s.nes_mut().peek(0x658) < bombs,
                    ActionKind::Boomerang => s.nes_mut().peek(0xBB) != 0 || c.frames > 3,
                    _ => c.frames > 3,
                };
                // A hit during the step (Manhandla's room) may knock Link out of his attack: not the mask's fault.
                assert!(
                    started || c.health_lost > 0.0,
                    "{name} step {step}: {a:?} did nothing ({} frames)",
                    c.frames
                );
                s.restore(&snap);
            }
        }
        let a = random_action(s.mask(), &mut rng);
        s.act(a).unwrap();
    }
}

/// F094: walking out of a cave (0x0a's white-sword cave) or a dungeon (0_37c) and loading the pre-entry state of
/// that exit give the same flag and mask: MOVE N back in is masked on the mouth, legal once Link left its tile.
#[test]
fn a_loaded_mouth_exit_matches_the_live_one() {
    let Some(nes) = open_core() else { return };
    let Some(nes2) = open_core() else { return };
    let map = Arc::new(GameMap::builtin());
    let (mut live, mut loaded) = (Session::new(nes, map.clone()), Session::new(nes2, map));
    let opts = LoadOptions::default();
    live.load(&common::load_named_state("0_0acave_0"), &opts, &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    live.set_capture_pre_entry(true);
    let north = Action::parse("MOVE N 16").unwrap();
    for _ in 0..20 {
        let c = live.act(Action::parse("MOVE S 16").unwrap()).unwrap();
        if c.changed_location() {
            break;
        }
    }
    assert!(!live.state().in_cave());
    assert!(live.just_exited_cave() && !live.mask().is_allowed(&north));
    let pre = live.last_pre_entry().expect("pre-entry captured").machine.clone();
    loaded.load(&pre, &LoadOptions { randomize_rng: false, ..opts.clone() }, &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert_eq!(loaded.state().full_location(), live.state().full_location());
    assert_eq!(loaded.just_exited_cave(), live.just_exited_cave());
    assert_eq!(loaded.mask(), live.mask());
    // Off the mouth tile (a spawn may cut the first steps short): MOVE N is legal again.
    let tile = loaded.state().link.tile();
    for _ in 0..8 {
        if loaded.state().link.tile() != tile {
            break;
        }
        loaded.act(Action::parse("MOVE S 4").unwrap()).unwrap();
    }
    assert_ne!(loaded.state().link.tile(), tile);
    assert!(!loaded.just_exited_cave());

    // A dungeon mouth (out of Dungeon 1 onto 0x37).
    loaded.load(&common::load_named_state("0_37c_0"), &opts, &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert!(loaded.just_exited_cave() && !loaded.mask().is_allowed(&north));
}

/// F177: an action whose weapon look-ahead runs (a beam) still leaves its last rendered frame as the screen.
#[test]
fn a_look_ahead_keeps_the_rendered_frame() {
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    s.set_render(true);
    s.load(&common::load_named_state("0_67s_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0))
        .unwrap();
    s.nes_mut().poke(0x66F, 0x22);
    s.nes_mut().poke(0x670, 0xFF);
    s.act(Action::parse("MOVE N 16").unwrap()).unwrap();
    assert!(s.nes_mut().pixels().is_some(), "rendered after a move");
    let beams = Action::parse("BEAMS N").unwrap();
    let fired = s.act(beams).unwrap();
    assert!(fired.frames > 3, "the sword swung");
    assert!(s.nes_mut().pixels().is_some(), "the action's last frame is still the screen");
}

/// N006: the attack directions follow the game's asm:Link_BorderBounds exactly. On a dungeon and an overworld state,
/// for every coordinate on each facing's axis, a 2-frame A (sword) or B (bomb) press starts the weapon iff
/// `sword_directions`/`item_directions` include the facing.
#[test]
fn attack_directions_match_the_game_at_every_coordinate() {
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    let dirs = [(Direction::N, 0x10u8, 8u8), (Direction::S, 0x20, 4), (Direction::W, 0x40, 2), (Direction::E, 0x80, 1)];
    for name in ["1_44w_0", "0_67s_0"] {
        s.load(&common::load_named_state(name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        for _ in 0..5 {
            s.act(Action::parse("WAIT 8").unwrap()).unwrap();
        }
        for slot in 1..12u16 {
            s.nes_mut().poke(0x34F + slot, 0);
        }
        s.nes_mut().poke(0x658, 8);
        let base = s.snapshot();
        for (d, button, face) in dirs {
            let vertical = matches!(d, Direction::N | Direction::S);
            for v in 0..=255u8 {
                for (b_item, press) in [(false, 0x01u8), (true, 0x02)] {
                    s.restore(&base);
                    let (x, y) = if vertical { (120, v) } else { (v, 141) };
                    for (addr, val) in [(0x70, x), (0x84, y), (0x394, 0), (0x53, 0), (0x98, face), (0xAC, 0), (0x656, 1)] {
                        s.nes_mut().poke(addr, val);
                    }
                    let view = s.apply_ram(&[]).clone();
                    let expected =
                        if b_item { view.item_directions() } else { view.sword_directions() }.contains(&d);
                    let bombs = s.nes_mut().peek(0x658);
                    s.press(button | press, 2);
                    let started = if b_item { s.nes_mut().peek(0x658) < bombs } else { s.nes_mut().peek(0xB9) != 0 };
                    assert_eq!(started, expected, "{name} {d:?} at {v} ({})", if b_item { "B" } else { "A" });
                }
            }
        }
    }
}

/// Sideways attacks at a dungeon doorway against the game, on all four sides (the N door of 1:0x33, S of 1:0x53, W
/// of 1:0x44, E of 1:0x72) and at 4:0x11's open S bomb wall (round 20's wall walk). At every position Link takes
/// from the entry spot into the room, out through the door frame a frame at a time and back in, an attack across
/// the doorway's axis is offered iff the game turns Link that way (holding only that direction turns him on a room
/// position of the same room) and the executor's turn is sound (after SWORD that way Link stands on a room
/// position of the same room). So on the room's edge (Y $5D or $BD, X $20 or $D0) with DoorwayDir clear it is
/// offered, on the first position in the doorway it is not.
#[test]
fn doorway_turns_match_the_game() {
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    let in_room = |x: u8, y: u8| (0x20..=0xD0).contains(&x) && (0x5D..=0xBD).contains(&y);
    let held = |d: Direction| match d {
        Direction::N => 0x10u8,
        Direction::S => 0x20,
        Direction::W => 0x40,
        _ => 0x80,
    };
    for (name, door) in [
        ("1_33n_0", Direction::N),
        ("1_53s_0", Direction::S),
        ("1_44w_0", Direction::W),
        ("1_72e_0", Direction::E),
        ("4_11s_0", Direction::S),
    ] {
        s.load(&common::load_named_state(name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        for slot in 1..12u16 {
            s.nes_mut().poke(0x34F + slot, 0);
        }
        let room = s.state().full_location();
        let ok = |s: &mut Session<ZCore>| {
            let (x, y) = (s.nes_mut().peek(0x70), s.nes_mut().peek(0x84));
            s.state().full_location() == room && s.nes_mut().peek(0x12) == 5 && in_room(x, y)
        };
        let still = |s: &mut Session<ZCore>| s.state().full_location() == room && s.nes_mut().peek(0x12) == 5;
        // The entry spot, one step toward the door from it (4:0x11: Y $BF with DoorwayDir clear), into the room,
        // out to the door and back in from deep in the doorway.
        let entry = s.snapshot();
        let mut spots = vec![entry.clone()];
        s.press(held(door), 1);
        if still(&mut s) {
            spots.push(s.snapshot());
        }
        s.restore(&entry);
        for _ in 0..24 {
            s.press(held(door.opposite()), 1);
            spots.push(s.snapshot());
        }
        let mut deep = None;
        for _ in 0..60 {
            s.press(held(door), 1);
            if !still(&mut s) {
                break;
            }
            spots.push(s.snapshot());
            let (x, y) = (s.nes_mut().peek(0x70) as i32, s.nes_mut().peek(0x84) as i32);
            let depth = match door {
                Direction::N => 0x5D - y,
                Direction::S => y - 0xBD,
                Direction::W => 0x20 - x,
                _ => x - 0xD0,
            };
            if deep.is_none() && depth >= 12 {
                deep = Some(s.snapshot());
            }
        }
        if let Some(deep) = deep {
            s.restore(&deep);
            for _ in 0..16 {
                s.press(held(door.opposite()), 1);
                spots.push(s.snapshot());
            }
        }
        let across = if matches!(door, Direction::N | Direction::S) { [Direction::E, Direction::W] } else { [Direction::N, Direction::S] };
        let (mut offered_on_edge, mut masked_in_doorway) = (0, 0);
        for spot in &spots {
            for d in across {
                s.restore(spot);
                let (x, y, doorway) = (s.nes_mut().peek(0x70), s.nes_mut().peek(0x84), s.nes_mut().peek(0x53));
                let at = format!("{name} {d:?} at ({x:02X},{y:02X}) $53={doorway} facing {}", s.nes_mut().peek(0x98));
                let offered = s.state().sword_directions().contains(&d);
                let mut game = false;
                for _ in 0..16 {
                    s.press(held(d), 1);
                    if s.nes_mut().peek(0x98) == d.value() && ok(&mut s) {
                        game = true;
                        break;
                    }
                    if !still(&mut s) {
                        break;
                    }
                }
                s.restore(spot);
                s.act(Action::parse(&format!("SWORD {}", d.name())).unwrap()).unwrap();
                let executor = ok(&mut s);
                assert_eq!(offered, game && executor, "{at}: game turns {game}, executor sound {executor}");
                let edge = matches!((door, x, y), (Direction::N, _, 0x5D) | (Direction::S, _, 0xBD) | (Direction::W, 0x20, _) | (Direction::E, 0xD0, _));
                offered_on_edge += (offered && edge && doorway == 0) as u32;
                masked_in_doorway += (!offered && !in_room(x, y)) as u32;
            }
        }
        assert!(offered_on_edge > 0 && masked_in_doorway > 0, "{name}: {offered_on_edge} {masked_in_doorway}");
    }
}
