//! `zelda::power_on` on zcore: the boot is the base movie's inputs, the delay alone moves the RNG, the
//! file-select screen waits as long as the longest delay, and only registrable names are taken.

mod common;

use common::{open_core, repo};
use zcore::ZCore;
use zelda::power_on::{self, IDLE_STATES, IDLE_STEP, MAX_DELAY};
use zelda::{Nes, ZeldaError};

/// zcore that logs every frame's buttons.
struct Logged(ZCore, Vec<u8>);

impl Nes for Logged {
    type State = zcore::MachineState;

    fn run_frame(&mut self, buttons: u8, render: bool) {
        assert!(!render, "the boot renders nothing");
        self.0.run_frame(buttons, render);
        self.1.push(buttons);
    }

    fn ram(&self) -> &[u8; 0x800] {
        self.0.ram()
    }

    fn prg_ram(&self) -> &[u8; 0x2000] {
        self.0.prg_ram()
    }

    fn poke(&mut self, _: u16, _: u8) {
        panic!("the boot writes no memory");
    }

    fn snapshot(&self) -> Self::State {
        self.0.snapshot()
    }

    fn restore(&mut self, state: &Self::State) {
        self.0.restore(state);
    }

    fn poke_state(_: &mut Self::State, _: u16, _: u8) {
        panic!("the boot writes no memory");
    }
}

fn booted(name: &str, delay: u32) -> Option<Logged> {
    let mut nes = Logged(open_core()?, Vec::new());
    let frames = power_on::boot(&mut nes, name, delay).unwrap_or_else(|e| panic!("{name:?} {delay}: {e}"));
    assert_eq!(frames as usize, nes.1.len());
    assert_eq!((nes.peek(0x12), nes.peek(0x10), nes.peek(0xEB)), (5, 0, 0x77), "{name:?} {delay}");
    Some(nes)
}

const RNG: std::ops::RangeInclusive<usize> = 0x18..=0x24;

/// Slot 1's name as the game keeps it (`$0638`: its character codes, digits 0-9, then A-Z from 0x0A, space 0x24).
fn registered(nes: &Logged) -> String {
    nes.ram()[0x638..0x640]
        .iter()
        .map(|&code| match code {
            0x00..=0x09 => (b'0' + code) as char,
            0x0A..=0x23 => (b'A' + code - 0x0A) as char,
            0x24 => ' ',
            _ => '?',
        })
        .collect()
}

#[test]
fn the_boot_is_the_base_movies_inputs() {
    let Some(nes) = booted("LINK", 0) else { return };
    let movie = movie::Movie::load(&repo().join("corpus/movies/power_on_to_start.tfmovie")).unwrap();
    assert!(movie.pokes.is_empty());
    assert_eq!(nes.1, movie.inputs);
    assert_eq!(registered(&nes), "LINK    ");
    // The same game whatever the case of the name.
    let lower = booted("link", 0).unwrap();
    assert!(lower.ram() == nes.ram() && lower.prg_ram() == nes.prg_ram());
}

#[test]
fn the_delay_alone_moves_the_rng() {
    let (Some(a), Some(b), Some(c)) = (booted("LINK", 37), booted("LINK", 37), booted("LINK", 38)) else {
        return;
    };
    assert!(a.ram() == b.ram() && a.prg_ram() == b.prg_ram(), "the same name and delay give the same machine");
    assert_ne!(a.ram()[RNG], c.ram()[RNG], "one more idle frame is another RNG phase");
    assert_eq!(c.1.len(), a.1.len() + 1);
    // The idle is on the file-select screen: the frames after TO_FILE_SELECT, before the first START of the name.
    let to_file_select: u32 = power_on::TO_FILE_SELECT.iter().map(|&(_, frames)| frames).sum();
    let idle = &c.1[to_file_select as usize..][..38];
    assert!(idle.iter().all(|&buttons| buttons == 0));
    assert_eq!(c.1[to_file_select as usize + 38], 0x08);
    // Another name is another game with the same RNG (the name takes other inputs: more or fewer frames).
    let zora = booted("ZORA 64", 37).unwrap();
    assert_eq!(registered(&zora), "ZORA 64 ");
    assert_ne!(zora.1.len(), a.1.len());
}

#[test]
fn the_file_select_screen_waits_out_the_longest_delay() {
    let Some(core) = open_core() else { return };
    let mut nes = Logged(core, Vec::new());
    power_on::to_file_select(&mut nes).unwrap();
    let mut idled = 0;
    while idled < MAX_DELAY {
        let frames = IDLE_STEP.min(MAX_DELAY - idled);
        power_on::idle(&mut nes, frames);
        idled += frames;
        assert_eq!(nes.peek(0x12), 1, "still on the file-select screen after {idled} idle frames");
    }
    power_on::register_and_start(&mut nes, "LINK").unwrap();
    assert_eq!((nes.peek(0x12), nes.peek(0x10), nes.peek(0xEB)), (5, 0, 0x77));
    assert_eq!(registered(&nes), "LINK    ");
    assert!(matches!(power_on::boot(&mut Logged(open_core().unwrap(), Vec::new()), "LINK", MAX_DELAY + 1),
                     Err(ZeldaError::Boot(reason)) if reason.contains("32766")));
}

#[test]
fn only_registrable_names_boot() {
    for (name, problem) in [
        ("ZELDA", "second quest"),
        ("zelda1", "second quest"),
        ("", "1 to 8 characters"),
        ("   ", "all spaces"),
        ("NINELETTR", "1 to 8 characters"),
        ("A-B", "only A-Z, 0-9 and spaces"),
    ] {
        match power_on::name_cells(name) {
            Err(ZeldaError::Boot(reason)) => assert!(reason.contains(problem), "{name:?}: {reason}"),
            other => panic!("{name:?}: {other:?}"),
        }
    }
    // A refused name runs no frame.
    if let Some(core) = open_core() {
        let mut nes = Logged(core, Vec::new());
        assert!(power_on::boot(&mut nes, "ZELDA", 0).is_err());
        assert!(nes.1.is_empty());
    }
    assert_eq!(power_on::name_cells("az09 ").unwrap(), [0, 25, 33, 42, 43]);
    // Every character the board offers a name lands in the save slot as itself.
    if let Some(nes) = booted("Q7 0X9", 0) {
        assert_eq!(registered(&nes), "Q7 0X9  ");
    }
}

#[test]
fn idle_states_are_named_by_their_frames() {
    assert_eq!([0, 1999, 2000, 30000, 32766].map(power_on::nearest_idle), [0, 0, 2000, 30000, 30000]);
    assert_eq!(power_on::nearest_idle(MAX_DELAY), IDLE_STATES * IDLE_STEP);
    assert_eq!(power_on::idle_state_name(2000), "power_on/idle_02000");
    assert_eq!(power_on::idle_state_name(30000), "power_on/idle_30000");
}
