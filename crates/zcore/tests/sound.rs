//! Sound on the real ROM: the title music plays, and listening changes nothing in the machine.

use std::sync::Arc;
use zcore::sound::{CPU_HZ, Synth};
use zcore::{Rom, ZCore, state_file};

const RATE: f64 = 48_000.0;

/// The ROM; missing panics (`test_rom`), or None with `TRIFORCE_ALLOW_NO_ROM=1`.
fn rom() -> Option<Arc<Rom>> {
    test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("the ROM is Zelda"))
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

/// Runs `frames` frames without input, appending their sound to `out`.
fn listen(nes: &mut ZCore, synth: &mut Synth, frames: usize, out: &mut Vec<f32>) {
    for _ in 0..frames {
        nes.run_frame(0, false);
        synth.run(nes.sound().expect("sound is on"), out);
    }
}

#[test]
fn the_title_screen_plays_music_and_listening_changes_nothing() {
    let Some(rom) = rom() else { return };
    let (mut heard, mut deaf) = (ZCore::power_on(rom.clone()), ZCore::power_on(rom.clone()));
    heard.set_sound(true);
    assert!(deaf.sound().is_none());
    let mut synth = Synth::new(RATE);
    let mut music = Vec::new();
    listen(&mut heard, &mut synth, 600, &mut music);
    for _ in 0..600 {
        deaf.run_frame(0, false);
    }
    assert!(state_file::machines_equal(&heard.snapshot(), &deaf.snapshot()));

    // Ten seconds of title music: as many samples as the frames took, loud, and not clipping.
    let expected = heard.cycles() as f64 * RATE / CPU_HZ;
    assert!((music.len() as f64 - expected).abs() <= 2.0, "{} samples for {expected}", music.len());
    let late = &music[music.len() / 2..];
    assert!(rms(late) > 0.02, "rms {}", rms(late));
    assert!(late.iter().all(|x| x.abs() < 0.999), "no clipping");

    // The same frames give the same samples.
    let mut again = ZCore::power_on(rom);
    again.set_sound(true);
    let (mut synth, mut second) = (Synth::new(RATE), Vec::new());
    listen(&mut again, &mut synth, 600, &mut second);
    assert!(music == second);
}

#[test]
fn a_synth_that_joins_late_picks_the_music_up() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom);
    nes.set_sound(true);
    let (mut whole, mut unheard) = (Synth::new(RATE), Vec::new());
    listen(&mut nes, &mut whole, 400, &mut unheard);
    // One synth has heard everything, one takes the machine over now, one starts knowing nothing.
    let (mut late, mut blank) = (Synth::new(RATE), Synth::new(RATE));
    late.resume(&nes);
    let (mut heard, mut joined, mut naive) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..240 {
        nes.run_frame(0, false);
        let frame = nes.sound().unwrap();
        whole.run(frame, &mut heard);
        late.run(frame, &mut joined);
        blank.run(frame, &mut naive);
    }
    // The waveforms differ in phase (a takeover restarts the oscillators), the loudness must not: from the
    // first quarter second on, where a synth that knows nothing still misses the notes being held.
    let loudness = |samples: &[f32]| -> Vec<f32> { samples.chunks_exact(12_000).map(rms).collect() };
    let (heard, joined, naive) = (loudness(&heard), loudness(&joined), loudness(&naive));
    assert!(heard.iter().all(|&level| level > 0.02), "{heard:?}");
    for (a, b) in heard.iter().zip(&joined) {
        assert!((0.75..1.33).contains(&(b / a)), "{heard:?} vs {joined:?}");
    }
    assert!(naive[0] < 0.75 * heard[0], "{naive:?} vs {heard:?}");
}
