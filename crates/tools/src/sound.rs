//! `tf replay sound`: a replay's sound as music and sound effects (`zcore::tracks`), written as WAV files, and
//! with `--check` the numbers that show the tracks are right (docs/architecture.md "Sound").
//!
//! The replay plays once.  Beside the machine run: the player's own sound (everything on: the game's sound,
//! `full.wav`), tracks split with the beep (`music.wav`, `effects.wav`) and without it (`effects-no-beep.wav`).
//! The checks compare, before every frame, each track's driver with the machine's driver state, listen for
//! silence in the effects track where the game plays no effect, compare the music with the game's sound over
//! every stretch of five seconds or more without an effect (a fresh pair of synths started at the stretch),
//! list where an effect borrowed a voice from the music, and time the tracks against the single synth.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ::zcore::tracks::{self, Mix, Track, Tracks, jingle};
use ::zcore::{Rom, ZCore};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

const RATE: f64 = 48_000.0;
/// Frames without an effect before the effects track must be silent (the output filters settle).
const SETTLE: u32 = 12;
/// The shortest stretch without an effect in which the music is compared with the game's sound (5 s).
const STRETCH: u32 = 300;
/// `asm:CurLevel`.
const CUR_LEVEL: usize = 0x10;
/// The voices an effect can borrow from a song: registers `$4000-$4003`, `$4004-$4007`, `$400C-$400F`.
const VOICES: [(&str, usize, std::ops::RangeInclusive<u8>); 3] =
    [("square 0", tracks::CUR_TUNE0, 0x00..=0x03), ("square 1", tracks::CUR_TUNE1, 0x04..=0x07), ("noise", tracks::CUR_EFFECT, 0x0C..=0x0F)];

/// A 16-bit PCM WAV file written as the samples come.
struct Wav {
    file: BufWriter<File>,
    samples: u32,
}

impl Wav {
    fn create(path: &Path) -> Result<Wav> {
        let mut file = BufWriter::new(File::create(path).with_context(|| path.display().to_string())?);
        file.write_all(&[0; 44])?;
        Ok(Wav { file, samples: 0 })
    }

    fn write(&mut self, samples: impl IntoIterator<Item = f32>) -> Result<()> {
        for sample in samples {
            self.file.write_all(&((sample.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())?;
            self.samples += 1;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        let data = self.samples * 2;
        let rate = RATE as u32;
        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&(36 + data).to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes()); // PCM
        header.extend_from_slice(&1u16.to_le_bytes()); // mono
        header.extend_from_slice(&rate.to_le_bytes());
        header.extend_from_slice(&(rate * 2).to_le_bytes());
        header.extend_from_slice(&2u16.to_le_bytes());
        header.extend_from_slice(&16u16.to_le_bytes());
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data.to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header)?;
        self.file.flush()?;
        Ok(())
    }
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0, |peak, x| peak.max(x.abs()))
}

/// Whether the game's driver plays, or is asked to play, anything the effects carry (as the frame starts).
/// `beep`: whether the low-hearts beep counts.
fn effect_active(ram: &[u8; 0x800], beep: bool) -> bool {
    let tune0 = if beep { ram[tracks::CUR_TUNE0] } else { ram[tracks::CUR_TUNE0] & !tracks::BEEP };
    let tune0_request = ram[tracks::TUNE0_REQUEST] & if beep { 0x7F } else { 0x3F };
    tune0 != 0
        || [tracks::CUR_TUNE1, tracks::CUR_EFFECT, tracks::CUR_SAMPLE, tracks::BACKGROUND_SAMPLE].iter().any(|&a| ram[a] != 0)
        || jingle(ram[tracks::CUR_SONG])
        || jingle(ram[tracks::SONG_REQUEST])
        || tune0_request != 0
        || ram[tracks::TUNE1_REQUEST] != 0
        || ram[tracks::EFFECT_REQUEST] & 0x7F != 0
        || ram[tracks::SAMPLE_REQUEST] != 0
}

/// Frames counted and frames that differed, with the first few differences.
#[derive(Default)]
struct Tally {
    frames: u64,
    differ: u64,
    examples: Vec<Value>,
}

impl Tally {
    fn add(&mut self, frame: u32, differ: Vec<usize>) {
        self.frames += 1;
        if !differ.is_empty() {
            self.differ += 1;
            if self.examples.len() < 5 {
                let bytes: Vec<String> = differ.iter().map(|a| format!("${a:03X}")).collect();
                self.examples.push(json!({"frame": frame, "bytes": bytes}));
            }
        }
    }

    fn json(&self) -> Value {
        json!({"frames": self.frames, "differ": self.differ, "examples": self.examples})
    }
}

fn differ(mine: &[u8; 0x800], ram: &[u8; 0x800], addrs: &[usize]) -> Vec<usize> {
    addrs.iter().copied().filter(|&a| mine[a] != ram[a]).collect()
}

/// A stretch in which an effect borrows a voice from a playing song.
struct Borrow {
    voice: usize,
    start: u32,
    frames: u32,
    effect: u8,
    /// Frames in which the machine's writes to the voice were exactly the effects driver's (the song wrote none).
    muted: u32,
    /// Frames in which the music driver wrote to the voice, and in which its last volume there was not zero.
    music_wrote: u32,
    music_sounding: u32,
}

#[derive(Default)]
struct Checks {
    song: Tally,
    effect_state: Tally,
    jingle: Tally,
    tune0_without_beep: Tally,
    beep_left: u64,
    quiet_run: u32,
    quiet_beep_run: u32,
    silent_frames: u64,
    silent_peak: f32,
    /// The frames of the loudest effects samples where silence was expected, and of the mirror's mismatches.
    silent_loud: Vec<(u32, f32)>,
    mismatched_frames: Vec<u32>,
    mismatched_detail: Vec<Value>,
    mismatched_seen: u64,
    beep_frames: u64,
    beep_peak_without: f32,
    beep_peak_with: f32,
    check_silence: bool,
    /// No song plays (the effects track should then be the game's sound), and how far it is from it.
    songless: bool,
    songless_frames: u64,
    songless_off: f32,
    songless_loud: Vec<Value>,
    check_beep: bool,
    /// The stretch now compared: its synths, first frame, length and largest difference.
    stretch: Option<(Tracks, u32, u32, f32)>,
    stretches: Vec<(u32, u32, f32)>,
    /// The first frame of each stretch where the music differs from the game's sound, and what differed.
    stretch_diffs: Vec<Value>,
    borrows: Vec<Borrow>,
    open: [Option<usize>; 3],
    music_volume: [u8; 3],
    fanfares: Vec<(u32, u8)>,
    triforce_tunes: Vec<u32>,
    beeps: Vec<(u32, u32)>,
    requests: [BTreeSet<u8>; 5],
    time_single: f64,
    time_live: f64,
    time_split: f64,
}

impl Checks {
    /// Before frame `frame`: the drivers against the machine's state, and what the frame should sound like.
    fn before(&mut self, frame: u32, nes: &ZCore, split: &Tracks, quiet: &Tracks) {
        let ram = nes.ram();
        let song = ram[tracks::CUR_SONG];
        let music = song != 0 && !jingle(song);
        fn mine(tracks: &Tracks, track: Track) -> &[u8; 0x800] {
            tracks.driver_ram(Some(track)).expect("the split tracks run their drivers")
        }
        if music {
            self.song.add(frame, differ(mine(split, Track::Music), ram, &tracks::SONG_TIME));
        }
        self.effect_state.add(frame, differ(mine(split, Track::Effects), ram, &tracks::EFFECT_STATE));
        if jingle(song) {
            // A jingle is one phrase: `asm:SongPhraseIndex` ($6C) belongs to the multi-phrase songs and keeps the
            // music's value in the game.
            let jingle_time: Vec<usize> = tracks::SONG_TIME.iter().copied().filter(|&a| a != 0x6C).collect();
            self.jingle.add(frame, differ(mine(split, Track::Effects), ram, &jingle_time));
        }
        let without = mine(quiet, Track::Effects);
        self.beep_left += u64::from(without[tracks::CUR_TUNE0] == tracks::BEEP);
        if !matches!(ram[tracks::CUR_TUNE0], 0 | tracks::BEEP) {
            let off = (without[tracks::CUR_TUNE0] == 0)
                || without[tracks::TUNE0_SCRIPT_OFFSET] != ram[tracks::TUNE0_SCRIPT_OFFSET];
            self.tune0_without_beep.add(frame, if off { vec![tracks::TUNE0_SCRIPT_OFFSET] } else { vec![] });
        }
        let active = effect_active(ram, true);
        self.check_silence = !active && self.quiet_run >= SETTLE;
        self.songless = ram[tracks::CUR_SONG] == 0 && ram[tracks::SONG_REQUEST] == 0;
        self.quiet_run = if active { 0 } else { self.quiet_run + 1 };
        let active_but_beep = effect_active(ram, false);
        self.check_beep = !active_but_beep && self.quiet_beep_run >= SETTLE && ram[tracks::CUR_TUNE0] == tracks::BEEP;
        self.quiet_beep_run = if active_but_beep { 0 } else { self.quiet_beep_run + 1 };

        // A stretch of music without effects: the music track must be the game's sound.
        if music && !active {
            if self.stretch.is_none() {
                let tracks = Tracks::new(nes, RATE, Mix { effects: false, ..Mix::ALL }, false, false);
                self.stretch = Some((tracks, frame, 0, 0.0));
            }
        } else {
            self.close_stretch();
        }
        if let Some((tracks, ..)) = &mut self.stretch {
            tracks.prepare(nes);
        }

        // Borrowed voices.
        for (voice, &(_, cur, _)) in VOICES.iter().enumerate() {
            let effect = ram[cur];
            let borrowed = music && effect != 0 && (voice < 2 || song & 0x91 != 0);
            match (borrowed, self.open[voice]) {
                (true, None) => {
                    self.open[voice] = Some(self.borrows.len());
                    self.borrows.push(Borrow { voice, start: frame, frames: 0, effect, muted: 0, music_wrote: 0, music_sounding: 0 });
                }
                (false, Some(_)) => self.open[voice] = None,
                _ => {}
            }
        }

        // What the replay asks for, for choosing stretches and the report.
        let request = |a: usize| ram[a];
        for (set, addr) in self.requests.iter_mut().zip(tracks::SONG_REQUEST..=tracks::TUNE0_REQUEST) {
            if request(addr) != 0 {
                set.insert(request(addr));
            }
        }
        match request(tracks::SONG_REQUEST) {
            0x08 => self.fanfares.push((frame, ram[CUR_LEVEL])),
            0x04 => self.triforce_tunes.push(frame),
            _ => {}
        }
        if ram[tracks::CUR_TUNE0] == tracks::BEEP || request(tracks::TUNE0_REQUEST) == tracks::BEEP {
            match self.beeps.last_mut() {
                Some((_, end)) if frame - *end < 300 => *end = frame,
                _ => self.beeps.push((frame, frame)),
            }
        }
    }

    fn close_stretch(&mut self) {
        if let Some((_, start, frames, off)) = self.stretch.take() {
            if frames >= STRETCH {
                self.stretches.push((start, frames, off));
            }
        }
    }

    /// After the frame ran.
    fn after(&mut self, frame: u32, nes: &ZCore, split: &Tracks, quiet: &Tracks) {
        let effects = split.samples(Track::Effects);
        if self.check_silence {
            if self.songless {
                // What still sounds here is the tail of an effect the game's driver has finished with: the full
                // mix must have it too.
                let game = split.samples(Track::Game);
                self.songless_frames += 1;
                let off = game.iter().zip(effects).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                self.songless_off = self.songless_off.max(off);
                if off > 1e-4 && self.songless_loud.len() < 10 {
                    self.songless_loud.push(json!([frame, off, peak(effects), peak(game)]));
                }
            }
            self.silent_frames += 1;
            self.silent_peak = self.silent_peak.max(peak(effects));
            if peak(effects) > 1e-4 && self.silent_loud.len() < 10 {
                self.silent_loud.push((frame, peak(effects)));
            }
        }
        if split.stats().mismatched > self.mismatched_seen {
            self.mismatched_seen = split.stats().mismatched;
            if self.mismatched_frames.len() < 16 {
                self.mismatched_frames.push(frame);
            }
            if self.mismatched_detail.len() < 4 {
                let mirror: Vec<String> =
                    split.driver_writes(None).iter().map(|w| format!("{}:40{:02X}={:02X}", w.cycle, w.register, w.value)).collect();
                let machine: Vec<String> = nes.sound().map_or(Vec::new(), |f| {
                    f.events().iter().map(|e| format!("{}:{:?}={:02X}", e.cycle(), e.register(), e.value())).collect()
                });
                let ram = nes.ram();
                self.mismatched_detail.push(json!({"frame": frame, "mirror": mirror, "machine": machine,
                    "after": format!("{:02X?} pause {:02X}", &ram[0x600..0x60A], ram[tracks::PAUSE_STATE])}));
            }
        }
        if self.check_beep {
            self.beep_frames += 1;
            self.beep_peak_with = self.beep_peak_with.max(peak(effects));
            self.beep_peak_without = self.beep_peak_without.max(peak(quiet.samples(Track::Effects)));
        }
        if let Some((tracks, start, frames, off)) = &mut self.stretch {
            let mut out = Vec::new();
            tracks.run(nes, &mut out);
            let game = tracks.samples(Track::Game);
            let worst = game.iter().zip(&out).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            if worst > 1e-6 && *off <= 1e-6 && self.stretch_diffs.len() < 12 {
                let rel = |w: &[tracks::Write]| -> Vec<(u32, u8, u8)> {
                    let at = w.first().map_or(0, |w| w.cycle);
                    w.iter().map(|w| (w.cycle - at, w.register, w.value)).collect()
                };
                let (mine, real) = (rel(tracks.driver_writes(Some(Track::Music))), rel(tracks.driver_writes(None)));
                let first = mine.iter().zip(&real).position(|(a, b)| a != b);
                let events: Vec<String> = nes.sound().map_or(Vec::new(), |f| {
                    f.events().iter().map(|e| format!("{}:{:?}={:02X}", e.cycle(), e.register(), e.value())).collect()
                });
                self.stretch_diffs.push(json!({"stretch": start, "frame": frame, "into": *frames, "worst": worst,
                    "writes_same": mine == real, "music_writes": mine.len(), "machine_writes": real.len(),
                    "first_differing_write": first.map(|i| format!("{:?} vs {:?}", mine.get(i), real.get(i))),
                    "machine_events": events}));
            }
            *off = off.max(worst);
            *frames += 1;
        }
        let real = split.driver_writes(None);
        let (music, fx) = (split.driver_writes(Some(Track::Music)), split.driver_writes(Some(Track::Effects)));
        for (voice, (_, _, registers)) in VOICES.iter().enumerate() {
            let on = |w: &&tracks::Write| registers.contains(&w.register);
            let mut wrote = false;
            for write in music.iter().filter(on) {
                wrote = true;
                if write.register == *registers.start() {
                    self.music_volume[voice] = if write.value & 0x10 == 0 { 0x0F } else { write.value & 0x0F };
                }
            }
            if let Some(index) = self.open[voice] {
                let borrow = &mut self.borrows[index];
                borrow.frames += 1;
                let same = real.iter().filter(on).map(|w| (w.register, w.value)).eq(fx.iter().filter(on).map(|w| (w.register, w.value)));
                borrow.muted += u32::from(same);
                borrow.music_wrote += u32::from(wrote);
                borrow.music_sounding += u32::from(self.music_volume[voice] != 0);
            }
        }
    }

    fn json(mut self, split: &Tracks) -> Value {
        self.close_stretch();
        let longest = self.stretches.iter().map(|s| s.1).max().unwrap_or(0);
        let off = self.stretches.iter().map(|s| s.2).fold(0.0f32, f32::max);
        let mut by_voice = Vec::new();
        for (voice, (name, ..)) in VOICES.iter().enumerate() {
            let all: Vec<&Borrow> = self.borrows.iter().filter(|b| b.voice == voice).collect();
            by_voice.push(json!({
                "voice": name,
                "stretches": all.len(),
                "frames": all.iter().map(|b| u64::from(b.frames)).sum::<u64>(),
                "machine_song_muted_frames": all.iter().map(|b| u64::from(b.muted)).sum::<u64>(),
                "music_wrote_frames": all.iter().map(|b| u64::from(b.music_wrote)).sum::<u64>(),
                "music_sounding_frames": all.iter().map(|b| u64::from(b.music_sounding)).sum::<u64>(),
            }));
        }
        let table: Vec<Value> = self
            .borrows
            .iter()
            .filter(|b| b.frames >= 8)
            .take(16)
            .map(|b| {
                json!({"frames": format!("{}-{}", b.start, b.start + b.frames - 1), "voice": VOICES[b.voice].0,
                       "effect": format!("${:02X}", b.effect), "machine_song_muted": format!("{}/{}", b.muted, b.frames),
                       "music_wrote": format!("{}/{}", b.music_wrote, b.frames),
                       "music_sounding": format!("{}/{}", b.music_sounding, b.frames)})
            })
            .collect();
        let stats = split.stats();
        let hex = |set: &BTreeSet<u8>| set.iter().map(|v| format!("${v:02X}")).collect::<Vec<_>>();
        let frames = stats.frames.max(1) as f64;
        json!({
            "driver": {"frames": stats.frames, "driven": stats.driven, "mirror_mismatched": stats.mismatched,
                       "mismatched_frames": self.mismatched_frames, "mismatched_detail": self.mismatched_detail,
                       "faults": stats.faults, "stray": stats.stray.iter().map(|a| format!("${a:04X}")).collect::<Vec<_>>()},
            "music_song_state": self.song.json(),
            "effects_state": self.effect_state.json(),
            "effects_jingle_state": self.jingle.json(),
            "effects_silence": {"frames": self.silent_frames, "peak": self.silent_peak,
                                "loudest_first": self.silent_loud.iter().map(|(f, p)| json!([f, p])).collect::<Vec<_>>(),
                                "songless_frames": self.songless_frames, "songless_max_abs_difference_from_full": self.songless_off,
                                "songless_differing_first": self.songless_loud},
            "music_is_the_game_without_effects": {"stretches": self.stretches.len(), "longest_frames": longest,
                "frames": self.stretches.iter().map(|s| u64::from(s.1)).sum::<u64>(), "max_abs_difference": off,
                "first": self.stretches.iter().take(5).map(|s| json!({"start": s.0, "frames": s.1, "max_abs_difference": s.2})).collect::<Vec<_>>(),
                "first_differences": self.stretch_diffs},
            "borrowed_voices": by_voice,
            "borrowed_table": table,
            "beep_off": {"beep_left_frames": self.beep_left, "other_tune0": self.tune0_without_beep.json(),
                         "beep_only_frames": self.beep_frames, "peak_with_beep": self.beep_peak_with,
                         "peak_without_beep": self.beep_peak_without},
            "requests_seen": {"song": hex(&self.requests[0]), "sample": hex(&self.requests[1]), "tune1": hex(&self.requests[2]),
                              "effect": hex(&self.requests[3]), "tune0": hex(&self.requests[4])},
            "moments": {"fanfares": self.fanfares.iter().map(|(f, level)| json!({"frame": f, "level": level})).collect::<Vec<_>>(),
                        "triforce_tunes": self.triforce_tunes, "beeps": self.beeps.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>()},
            "microseconds_per_frame": {"single_synth": 1e6 * self.time_single / frames,
                                       "live_tracks_warm": 1e6 * self.time_live / frames,
                                       "split_tracks": 1e6 * self.time_split / frames},
        })
    }
}

/// Plays `replay` from frame `from` for `frames` frames (to its end by default), writing `full.wav`,
/// `music.wav`, `effects.wav` and `effects-no-beep.wav` into `out` if given; with `check` the numbers.
pub fn sound(
    rom: Arc<Rom>,
    replay: tfreplay::Replay,
    resolve: &dyn Fn(&str) -> std::io::Result<Vec<u8>>,
    out: Option<&Path>,
    from: u32,
    frames: Option<u32>,
    check: bool,
) -> Result<Value> {
    let mut player = tfreplay::Player::open(rom, replay, resolve)?;
    ensure!(from <= player.frames(), "frame {from} is past the replay's {} frames", player.frames());
    let end = frames.map_or(player.frames(), |n| (from + n).min(player.frames()));
    player.seek(from)?;
    player.set_sound(Some(RATE));
    let nes = player.nes();
    let mut split = Tracks::new(nes, RATE, Mix::ALL, true, false);
    let mut quiet = Tracks::new(nes, RATE, Mix { beep: false, ..Mix::ALL }, true, false);
    // The same sound made as play-zelda makes it (every track kept ready) and as a single synth.
    let mut live = Tracks::new(nes, RATE, Mix::ALL, false, true);
    let mut single = Tracks::new(nes, RATE, Mix::ALL, false, false);
    let mut wavs = Vec::new();
    if let Some(dir) = out {
        std::fs::create_dir_all(dir)?;
        for name in ["full.wav", "music.wav", "effects.wav", "effects-no-beep.wav"] {
            wavs.push(Wav::create(&dir.join(name))?);
        }
    }
    let mut checks = Checks::default();
    let (mut full_hash, mut single_hash) = (sha1_smol::Sha1::new(), sha1_smol::Sha1::new());
    let (mut pairs, mut quiet_pairs, mut scratch) = (Vec::new(), Vec::new(), Vec::new());
    while player.frame() < end {
        let frame = player.frame();
        let ran = player.step_watched(false, &mut |nes, after| {
            if !after {
                split.prepare(nes);
                quiet.prepare(nes);
                if check {
                    let t = Instant::now();
                    single.prepare(nes);
                    checks.time_single += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    live.prepare(nes);
                    checks.time_live += t.elapsed().as_secs_f64();
                    checks.before(frame, nes, &split, &quiet);
                }
                return;
            }
            let t = Instant::now();
            split.run(nes, &mut pairs);
            checks.time_split += t.elapsed().as_secs_f64();
            quiet.run(nes, &mut quiet_pairs);
            if check {
                scratch.clear();
                let t = Instant::now();
                single.run(nes, &mut scratch);
                checks.time_single += t.elapsed().as_secs_f64();
                for sample in &scratch {
                    single_hash.update(&sample.to_le_bytes());
                }
                scratch.clear();
                let t = Instant::now();
                live.run(nes, &mut scratch);
                checks.time_live += t.elapsed().as_secs_f64();
                checks.after(frame, nes, &split, &quiet);
            }
        })?;
        if !ran {
            break;
        }
        let full = std::mem::take(player.audio());
        for sample in &full {
            full_hash.update(&sample.to_le_bytes());
        }
        if let [full_wav, music, effects, no_beep] = &mut wavs[..] {
            full_wav.write(full.iter().copied())?;
            music.write(pairs.iter().step_by(2).copied())?;
            effects.write(pairs.iter().skip(1).step_by(2).copied())?;
            no_beep.write(quiet_pairs.iter().skip(1).step_by(2).copied())?;
        }
        pairs.clear();
        quiet_pairs.clear();
    }
    for wav in wavs {
        wav.finish()?;
    }
    let mut report = json!({"from": from, "to": player.frame(), "rate": RATE, "full_sha1": full_hash.digest().to_string()});
    if check {
        report["single_synth_sha1"] = json!(single_hash.digest().to_string());
        report["checks"] = checks.json(&split);
    }
    Ok(report)
}
