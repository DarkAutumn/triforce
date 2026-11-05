//! The same sessions on zcore and on the FCEUmm oracle: identical RAM after every action.
#![cfg(feature = "zcore")]

mod common;

use common::{OracleNes, open_oracle, repo};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::actions::{NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS};
use zelda::{Action, GameMap, LoadOptions, Nes, Session};

struct Play<'a, N: Nes>(&'a mut N);

impl<N: Nes> movie::Playback for Play<'_, N> {
    fn run_frame(&mut self, buttons: u8) {
        self.0.run_frame(buttons, false);
    }
    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value);
    }
}

fn movies() -> std::path::PathBuf {
    repo().join("corpus/movies")
}

fn zcore() -> zcore::ZCore {
    let bytes = std::fs::read(repo().join("roms/zelda.nes")).expect("roms/zelda.nes");
    zcore::ZCore::power_on(zcore::Rom::from_ines(&bytes).expect("Zelda ROM"))
}

/// Plays `power_on_to_start` on `nes` and returns the machine state at its end.
fn base<N: Nes>(nes: &mut N) -> N::State {
    let m = movie::Movie::load(&movies().join("power_on_to_start.tfmovie")).unwrap();
    movie::play(&mut Play(nes), &m, &movies()).unwrap();
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
fn zcore_matches_oracle_over_random_play() {
    let Some(mut o) = open_oracle() else { return };
    let mut z = zcore();
    let (ob, zb) = (base(&mut o), base(&mut z));
    assert_eq!(o.ram(), z.ram(), "base state RAM");
    let map = Arc::new(GameMap::builtin());
    let mut os: Session<OracleNes> = Session::new(o, map.clone());
    let mut zs = Session::new(z, map);
    let opts = LoadOptions { randomize_rng: true, ..LoadOptions::default() };
    for seed in 0..4u64 {
        os.load(&ob, &opts, &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
        zs.load(&zb, &opts, &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
        assert_eq!(os.nes().ram(), zs.nes().ram(), "seed {seed}: after load");
        let mut rng = ChaCha8Rng::seed_from_u64(100 + seed);
        for step in 0..300 {
            if zs.state().game_over() {
                break;
            }
            assert_eq!(os.mask().to_flat(), zs.mask().to_flat(), "seed {seed} step {step}: mask");
            let a = random_action(zs.mask(), &mut rng);
            let (oc, zc) = (os.act(a), zs.act(a));
            assert_eq!(format!("{oc:?}"), format!("{zc:?}"), "seed {seed} step {step}: change for {a:?}");
            assert_eq!(os.nes().ram(), zs.nes().ram(), "seed {seed} step {step}: RAM after {a:?}");
            assert_eq!(os.nes().prg_ram(), zs.nes().prg_ram(), "seed {seed} step {step}: PRG-RAM");
            if zc.is_err() {
                break;
            }
        }
        assert_eq!(os.frame(), zs.frame(), "seed {seed}: frames");
    }
}

#[test]
fn zcore_snapshot_restore_is_exact() {
    let mut z = zcore();
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
