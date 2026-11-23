//! Music and sound effects apart (`zcore::tracks`) on the real ROM: the title music, then the file menu's
//! tunes.

use std::sync::Arc;
use zcore::sound::Synth;
use zcore::tracks::{Mix, Track, Tracks};
use zcore::{Rom, ZCore};

const RATE: f64 = 48_000.0;
const START: u8 = 0x08;
const SELECT: u8 = 0x04;

/// The ROM; missing panics (`test_rom`), or None with `TRIFORCE_ALLOW_NO_ROM=1`.
fn rom() -> Option<Arc<Rom>> {
    test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("the ROM is Zelda"))
}

/// The buttons of frame `f`: ten seconds of title music, Start, then Select every half second on the file menu
/// (each move plays a tune on square 1).
fn buttons(f: usize) -> u8 {
    match f {
        600 => START,
        f if f >= 760 && f < 1060 && (f - 760) % 30 == 0 => SELECT,
        _ => 0,
    }
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0, |peak, x| peak.max(x.abs()))
}

#[test]
fn the_tracks_follow_the_game_and_change_nothing_with_everything_on() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom);
    nes.set_sound(true);
    let mut synth = Synth::new(RATE);
    let mut all = Tracks::new(&nes, RATE, Mix::ALL, false, false);
    let mut split = Tracks::new(&nes, RATE, Mix::ALL, true, false);
    let (mut plain, mut through, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut timed, mut title_music_off_by, mut title_effects_peak) = (0, 0.0f32, 0.0f32);
    let mut menu_effects_peak = 0.0f32;
    for f in 0..1400 {
        all.prepare(&nes);
        split.prepare(&nes);
        nes.run_frame(buttons(f), false);
        let frame = nes.sound().expect("sound is on");
        synth.run(frame, &mut plain);
        all.run(&nes, &mut through);
        split.run(&nes, &mut pairs);

        // The mirror driver wrote what the machine's did, at the same cycles from its first write (no sample
        // plays here, so no DMC fetch moves the machine's CPU).
        let mirror = split.driver_writes(None);
        if let Some(first) = mirror.first() {
            let real: Vec<_> = frame.events().iter().filter(|e| e.register().is_some()).take(mirror.len()).collect();
            let at = real[0].cycle();
            for (write, event) in mirror.iter().zip(real) {
                assert_eq!(write.cycle - first.cycle, event.cycle() - at, "frame {f}");
                assert_eq!(0x4000 + u16::from(write.register), event.register().unwrap(), "frame {f}");
            }
            timed += 1;
        }
        let (game, music, effects) =
            (split.samples(Track::Game), split.samples(Track::Music), split.samples(Track::Effects));
        assert_eq!((music.len(), effects.len()), (game.len(), game.len()));
        if f < 600 {
            let off = game.iter().zip(music).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            title_music_off_by = title_music_off_by.max(off);
            title_effects_peak = title_effects_peak.max(peak(effects));
        } else {
            menu_effects_peak = menu_effects_peak.max(peak(effects));
        }
    }
    // With everything on the sound is the machine's writes through one synth, sample for sample.
    assert!(plain == through, "{} vs {} samples", plain.len(), through.len());
    assert_eq!(pairs.len(), 2 * plain.len());
    let stats = split.stats();
    assert!(stats.driven > 1000 && timed as u64 == stats.driven, "{stats:?}, {timed} timed");
    assert_eq!((stats.mismatched, stats.faults), (0, 0), "{stats:?}");
    assert!(stats.stray.is_empty(), "the driver touched {:04X?}", stats.stray);
    // The title song alone: the music track is the game's sound; the effects track is silent.
    assert_eq!(title_music_off_by, 0.0);
    assert!(title_effects_peak < 1e-6, "{title_effects_peak}");
    assert!(menu_effects_peak > 0.01, "the menu's tunes are effects: {menu_effects_peak}");
}

#[test]
fn switching_while_it_plays_keeps_the_sample_clock() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom);
    nes.set_sound(true);
    let mut synth = Synth::new(RATE);
    let mut live = Tracks::new(&nes, RATE, Mix::ALL, false, true);
    let (mut plain, mut out) = (Vec::new(), Vec::new());
    let mixes = [
        Mix { music: false, ..Mix::ALL },
        Mix { effects: false, ..Mix::ALL },
        Mix { beep: false, ..Mix::ALL },
        Mix { music: false, effects: false, beep: true },
        Mix::ALL,
    ];
    for f in 0..1400 {
        if f % 250 == 125 {
            live.set_mix(mixes[(f / 250) % mixes.len()]);
        }
        live.prepare(&nes);
        nes.run_frame(buttons(f), false);
        synth.run(nes.sound().unwrap(), &mut plain);
        live.run(&nes, &mut out);
    }
    assert_eq!(out.len(), plain.len());
    let stats = live.stats();
    assert_eq!((stats.mismatched, stats.faults), (0, 0), "{stats:?}");
}
