//! `zelda::power_on` on zcore: the boot is the base movie's inputs, the delay alone moves the RNG, the
//! file-select screen waits as long as the longest delay, only registrable names are taken, and the frame rule
//! is 4 frames (a run's delay is its index in frame rules).

mod common;

use common::{open_core, repo};
use zcore::ZCore;
use zelda::executor::buttons::{LEFT, UP};
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

/// A person may type ZELDA (`emu-zelda.py --name ZELDA`): the board takes it (`board_cells`, which a run's
/// `name_cells` refuses), and the game starts slot 1 in the second quest (`QuestNumbers` $062D = 1).
#[test]
fn zelda_is_typed_and_starts_the_second_quest() {
    assert!(power_on::name_cells("ZELDA").is_err());
    assert_eq!(power_on::board_cells("zelda").unwrap(), [25, 4, 11, 3, 0]);
    let Some(core) = open_core() else { return };
    let mut nes = Logged(core, Vec::new());
    power_on::to_file_select(&mut nes).unwrap();
    power_on::register_and_start(&mut nes, "ZELDA").unwrap();
    assert_eq!((nes.peek(0x12), nes.peek(0x10), nes.peek(0xEB)), (5, 0, 0x77));
    assert_eq!(registered(&nes), "ZELDA   ");
    assert_eq!(nes.peek(0x062D), 1, "the second quest");
}

#[test]
fn idle_states_are_named_by_their_frames() {
    assert_eq!([0, 1999, 2000, 30000, 32766].map(power_on::nearest_idle), [0, 0, 2000, 30000, 30000]);
    assert_eq!(power_on::nearest_idle(MAX_DELAY), IDLE_STATES * IDLE_STEP);
    assert_eq!(power_on::idle_state_name(2000), "power_on/idle_02000");
    assert_eq!(power_on::idle_state_name(30000), "power_on/idle_30000");
}

/// Walks Link from the first gameplay frame into the sword cave (up, left, up onto the entrance) and waits out
/// the walk down its steps, until the cave mode ($0B) begins.  Returns the frames run.
fn into_the_sword_cave<N: Nes>(nes: &mut N) -> u32 {
    let mut frames = 0;
    let mut walk = |nes: &mut N, buttons: u8, more: &dyn Fn(&N) -> bool| {
        for _ in 0..200 {
            if !more(nes) {
                return;
            }
            nes.run_frame(buttons, false);
            frames += 1;
        }
        panic!("Link is stuck walking {buttons:#04x}");
    };
    walk(nes, UP, &|nes| nes.peek(0x84) > 93);
    walk(nes, LEFT, &|nes| nes.peek(0x70) > 64);
    walk(nes, UP, &|nes| nes.peek(0x12) == 5);
    assert_eq!(nes.peek(0x12), 0x10, "the walk down the cave's steps");
    walk(nes, 0, &|nes| nes.peek(0x12) == 0x10);
    assert_eq!(nes.peek(0x12), 0x0B, "in the cave");
    frames
}

/// The frame rule (`power_on::FRAME_RULE`, 4).  Every delay is another machine at the first controllable
/// frame, but the walk down the first cave's steps moves Link only on frames with `FrameCounter AND 3 = 0`
/// (asm:Mode10Stairs_Update_Full), so the delays fall in classes of exactly 4 consecutive ones that reach the
/// cave on the same frame since power on, with the same random numbers and `FrameCounter`; the next class
/// reaches it 4 frames later, another RNG phase; each class holds one multiple of 4.  Checked at the start of
/// the cycle, around the corpus golden's delay (16610) and at its end.
#[test]
fn the_frame_rule_is_four_frames() {
    let Some(mut nes) = open_core() else { return };
    assert_eq!(power_on::FRAME_RULE, 4);
    let to_file_select = power_on::to_file_select(&mut nes).unwrap();
    let mut at = 0;
    for delays in [0..24, 16_600..16_624, MAX_DELAY - 23..MAX_DELAY + 1] {
        power_on::idle(&mut nes, delays.start - at);
        at = delays.start;
        let file_select = nes.snapshot();
        // (delay, the first controllable frame's RNG and FrameCounter, frames since power on at the cave, its RNG
        // and FrameCounter there)
        let mut arrivals = Vec::new();
        for delay in delays.clone() {
            nes.restore(&file_select);
            power_on::idle(&mut nes, delay - delays.start);
            let started = power_on::register_and_start(&mut nes, "LINK").unwrap();
            let start = (nes.ram()[RNG].to_vec(), nes.peek(0x15));
            let frame = to_file_select + delay + started + into_the_sword_cave(&mut nes);
            arrivals.push((delay, start, frame, nes.ram()[RNG].to_vec(), nes.peek(0x15)));
        }
        nes.restore(&file_select);
        let mut classes: Vec<Vec<u32>> = Vec::new();
        for (i, (delay, start, frame, rng, counter)) in arrivals.iter().enumerate() {
            let Some((_, last_start, last_frame, last_rng, last_counter)) = i.checked_sub(1).map(|i| &arrivals[i])
            else {
                classes.push(vec![*delay]);
                continue;
            };
            assert_ne!(start, last_start, "delay {delay}: another machine at the first controllable frame");
            if frame == last_frame {
                assert!((rng, counter) == (last_rng, last_counter), "delay {delay}: the same time is the same RNG");
                classes.last_mut().unwrap().push(*delay);
            } else {
                assert_eq!(*frame, last_frame + power_on::FRAME_RULE, "delay {delay}: the next frame rule");
                assert_ne!(rng, last_rng, "delay {delay}: another frame rule is another RNG phase");
                classes.push(vec![*delay]);
            }
        }
        let inner = &classes[1..classes.len() - 1];
        assert!(inner.len() >= 4 && inner.iter().all(|class| class.len() == 4), "{classes:?}");
        assert!(inner.iter().all(|class| class.iter().filter(|&&d| d % 4 == 0).count() == 1), "{classes:?}");
    }
}

/// A power-on run's delay without a fixed one: its index (the run seed) in frame rules, 0, 4, 8, ...; the
/// delays start again at 0 after the 8192 frame rules of the RNG's period.
#[test]
fn a_runs_delay_is_its_index_in_frame_rules() {
    assert_eq!(power_on::FRAME_RULES, 8192);
    assert_eq!([0, 1, 2, 249, 1499, 8191].map(power_on::run_delay), [0, 4, 8, 996, 5996, 32764]);
    assert_eq!([8192, 8193, 8192 * 3 + 5].map(power_on::run_delay), [0, 4, 20]);
    assert!((0..20_000).all(|index| power_on::run_delay(index) <= MAX_DELAY));
}
