//! Sessions on zcore from the base movie's end: snapshot/restore replays random play exactly.
#![cfg(feature = "zcore")]

mod common;

use common::{open_core, repo};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zcore::ZCore;
use zelda::actions::{NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS};
use zelda::{Action, GameMap, LoadOptions, Nes, Session};

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
    for (front, expect_damage) in [(true, false), (false, true)] {
        s.load(&machine, &opts, &mut ChaCha8Rng::seed_from_u64(3)).unwrap();
        s.press(0, 120);
        let state = s.state().clone();
        let darknut = state
            .enemies
            .iter()
            .find(|e| matches!(e.id, 0x0B | 0x0C) && e.spawn_state == 0)
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
        if s.nes_mut().peek(0x394) == 0 && state.mode() == 5 {
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
