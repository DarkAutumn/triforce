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
