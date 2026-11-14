//! `zelda::rng` against the game on zcore: the function is the register the game holds after N frames from
//! reset, and a load that randomizes the RNG writes a natural state before any frame runs after the restore, so
//! every frame from the room entry on (the spawn clouds, the first monster decisions) runs on values the game can
//! hold.
#![cfg(feature = "zcore")]

mod common;

use common::{load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zcore::{MachineState, ZCore};
use zelda::rng::{self, LEAD_IN, LEN, PERIOD, PHASES};
use zelda::{GameMap, LoadOptions, Nes, Session};

/// The game's steps: the register on every frame from power on matches `rng::natural` of a step count that
/// starts when reset seeds it and grows by one per frame (by none on a lag frame, which has no NMI).
#[test]
fn the_register_is_the_games_from_power_on() {
    let Some(mut nes) = open_core() else { return };
    // Before reset has run the register is the power-on RAM, not yet seeded.
    let mut frames = 0u32;
    while rng::read(&nes) != rng::SEED {
        nes.run_frame(0, false);
        frames += 1;
        assert!(frames < 60, "reset seeds the register within a second");
    }
    let (mut steps, mut cur) = (0u32, rng::SEED);
    let mut at = std::collections::BTreeMap::new();
    let wanted = [0, LEAD_IN - 1, LEAD_IN, *PHASES.end(), *PHASES.end() + 1];
    while steps <= *PHASES.end() + 1 {
        let reg = rng::read(&nes);
        if reg != cur {
            (steps, cur) = (steps + 1, rng::step(&cur));
            assert_eq!(reg, cur, "frame {frames}: the register is {steps} steps from reset");
        }
        at.entry(steps).or_insert((frames, reg));
        nes.run_frame(0, false);
        frames += 1;
    }
    for n in wanted {
        let (frame, reg) = at[&n];
        assert_eq!(reg, rng::natural(n), "step {n} (frame {frame})");
    }
    // The lead-in never comes back; the cycle does, PERIOD steps on.
    assert_eq!(at[&(LEAD_IN + PERIOD)].1, at[&LEAD_IN].1);
    assert_ne!(at[&(LEAD_IN - 1)].1, rng::natural(LEAD_IN - 1 + PERIOD));
    assert_eq!(rng::phase_of(&at[&(LEAD_IN + PERIOD)].1), Some(LEAD_IN));
}

/// zcore that logs the RNG register at the start of every frame, the restores and the RNG pokes.
struct Watch {
    nes: ZCore,
    log: Vec<Event>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    Restore,
    /// A write into `$18-$24`.
    Poke,
    /// The register a frame starts with.
    Frame([u8; LEN]),
}

impl Nes for Watch {
    type State = MachineState;

    fn run_frame(&mut self, buttons: u8, render: bool) {
        self.log.push(Event::Frame(rng::read(&self.nes)));
        self.nes.run_frame(buttons, render);
    }

    fn ram(&self) -> &[u8; 0x800] {
        self.nes.ram()
    }

    fn prg_ram(&self) -> &[u8; 0x2000] {
        self.nes.prg_ram()
    }

    fn poke(&mut self, addr: u16, value: u8) {
        if (rng::ADDR..rng::ADDR + LEN as u16).contains(&addr) {
            self.log.push(Event::Poke);
        }
        self.nes.poke(addr, value);
    }

    fn snapshot(&self) -> MachineState {
        self.nes.snapshot()
    }

    fn restore(&mut self, state: &MachineState) {
        self.log.push(Event::Restore);
        self.nes.restore(state);
    }

    fn poke_state(state: &mut MachineState, addr: u16, value: u8) {
        state.poke(addr, value);
    }
}

/// Every frame's register from the log's last restore: all 13 RNG pokes come before the first frame, and each
/// frame starts on the cycle, one step (or, on a lag frame, none) after the frame before.  Returns the frames.
fn natural_from_the_restore(log: &[Event], phase: u32) -> usize {
    let start = log.iter().rposition(|e| *e == Event::Restore).expect("a restore");
    let after = &log[start + 1..];
    let first_frame = after.iter().position(|e| matches!(e, Event::Frame(_))).expect("frames ran");
    let pokes = after.iter().filter(|e| **e == Event::Poke).count();
    assert_eq!(pokes, LEN, "the 13 bytes are written once");
    assert!(after[..first_frame].iter().all(|e| *e == Event::Poke), "every write before the first frame");
    let regs: Vec<[u8; LEN]> = after.iter().filter_map(|e| if let Event::Frame(r) = e { Some(*r) } else { None }).collect();
    assert_eq!(regs[0], rng::natural(phase), "the first frame runs on the written phase");
    let mut prev = None;
    for (k, reg) in regs.iter().enumerate() {
        let n = rng::phase_of(reg).filter(|n| PHASES.contains(n));
        let n = n.unwrap_or_else(|| panic!("frame {k} after the restore: {reg:02X?} is off the cycle"));
        if let Some(p) = prev {
            let next = if p == *PHASES.end() { *PHASES.start() } else { p + 1 };
            assert!(n == p || n == next, "frame {k}: phase {n} after {p}");
        }
        prev = Some(n);
    }
    regs.len()
}

fn session() -> Option<Session<Watch>> {
    let nes = open_core()?;
    Some(Session::new(Watch { nes, log: Vec::new() }, Arc::new(GameMap::builtin())))
}

/// A load with `randomize_rng` (a start state: saved pre-entry, the frame before the room entry) writes the
/// natural state right after the restore; the entry frames, the spawn clouds and 120 frames of play all run on
/// the cycle.  Overworld (0x3c from the cave, and the edge-spawn room 0x58), dungeon 2 and a boss room.
#[test]
fn a_randomized_load_runs_every_frame_on_the_cycle() {
    let Some(mut s) = session() else { return };
    let opts = LoadOptions { randomize_rng: true, fresh_room: true, ..Default::default() };
    for name in ["0_3cc_0", "0_58n_0", "2_7ds_0", "2_0de_0"] {
        let state = load_named_state(name);
        let mut phases = Vec::new();
        for seed in [1, 1, 2] {
            s.nes_mut().log.clear();
            s.load(&state, &opts, &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
            assert_eq!(s.nes().peek(0x12), 5, "{name}: playable after the load");
            s.press(0, 120);
            let phase = s.rng_phase().expect("a randomized load writes a phase");
            let frames = natural_from_the_restore(&s.nes().log, phase);
            assert!(frames > 120, "{name}: the entry frames and 120 of play ({frames})");
            phases.push(phase);
        }
        assert_eq!(phases[0], phases[1], "{name}: the same seed, the same phase");
        assert_ne!(phases[0], phases[2], "{name}: another seed, another phase");
    }
    // Without `randomize_rng` nothing is written and no phase is reported.
    s.nes_mut().log.clear();
    s.load(&load_named_state("0_3cc_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert_eq!(s.rng_phase(), None);
    assert!(!s.nes().log.contains(&Event::Poke));
}

/// `write_rng` writes the register `phase` steps after reset at the current frame and reports the phase; a
/// restore is no start and reports none.
#[test]
fn write_rng_writes_the_phase() {
    let Some(mut s) = session() else { return };
    let state = load_named_state("0_3cc_0");
    s.load(&state, &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    s.write_rng(1234);
    assert_eq!(rng::read(s.nes()), rng::natural(1234));
    assert_eq!(s.rng_phase(), Some(1234));
    let snapshot = s.snapshot();
    s.restore(&snapshot);
    assert_eq!(s.rng_phase(), None, "a restore is no start");
}
