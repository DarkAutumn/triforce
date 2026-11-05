//! Sessions on zcore from the base movie's end: snapshot/restore replays random play exactly.
#![cfg(feature = "zcore")]

mod common;

use common::{open_core, repo};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zcore::ZCore;
use zelda::actions::{NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS};
use zelda::{Action, GameMap, LoadOptions, Session};

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
