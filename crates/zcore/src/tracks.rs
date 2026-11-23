//! The game's sound in two tracks, music and sound effects, each of which can be switched off, and the
//! low-hearts beep on its own switch.
//!
//! The game's sound driver (`asm:Audio_Drive`, bank 0) runs once a frame from the NMI: it takes the five
//! request bytes the game wrote (`asm:SongRequest` .. `asm:Tune0Request`, `$600-$604`), drives tune 1, the
//! noise effect, the DMC sample, the song and tune 0, and zeroes the requests.  An effect borrows a voice:
//! while a square tune plays (`asm:CurTune0`, `asm:CurTune1`) the song keeps time but writes nothing to that
//! square.  The machine's register writes therefore cannot be sorted into music and effects without leaving
//! holes in the music.  Instead the driver's own code runs again beside the machine: each [`Driver`] is a
//! private copy of the driver's RAM, run on a small 6502 over the ROM's bank 0 and fixed bank whenever the
//! machine's driver ran, fed only its own requests and heard through its own [`Synth`]:
//!
//! - the music driver gets the songs (overworld, dungeon, level 9, Ganon, Zelda's song, the ending, the
//!   title) and never a tune, noise effect or sample; a jingle's request or a silence (`Tune0Request` `$80`,
//!   `Tune1Request` bit 7) stops its song as it stops the game's (`asm:Song_Silence`);
//! - the effects driver gets both square tunes, the noise effects, the DMC samples and the two jingles the
//!   game plays through its song driver: the item fanfare (`SongRequest` `$08`) and the end-of-level tune
//!   (`$04`); a song's request only stops a jingle that is playing;
//! - with the beep off, the low-hearts beep's request (`Tune0Request` `$40`) is dropped, and a beep that is
//!   playing ends at once, as its script's end would end it (`asm:Audio_DriveTune0`).
//!
//! Every driver is fed from the machine's RAM as it stands before the frame: the NMI, and with it the
//! driver, runs at the start of a frame, before the game's code that writes requests (`asm:Nmi_Isr`).  A
//! mirror driver gets every request; whatever byte of the driver's state the machine holds and the mirror
//! does not was written by the game itself (requests, `asm:Audio_SilenceAll` clearing the tunes, a reset)
//! and goes to every driver.  The drivers' writes are timed by the machine driver's first register write of
//! the frame; the machine's writes after its driver (the game's own, e.g. on unpausing) go to every synth,
//! and the DMC's sample bytes to the synths that play samples.
//!
//! With everything on, the sound is the machine's own writes through one synth ([`Track::Game`]), exactly
//! what it was before this module: the console mixes its voices non-linearly, so two tracks added up are
//! not the mix.  With the beep off and both tracks on, a driver fed everything but the beep plays
//! ([`Track::Quiet`]).  Nothing here changes the machine.

use std::sync::Arc;

use crate::sound::{Event, Frame, SAMPLE, Synth, Voices};
use crate::{Rom, ZCore};

/// `asm:Audio_Drive` in bank 0.
const AUDIO_DRIVE: u16 = 0x9825;
/// `asm:Song_Silence` in bank 0.
const SONG_SILENCE: u16 = 0x9D46;
/// `asm:PauseState`: the one byte outside its own state the driver reads.
pub const PAUSE_STATE: usize = 0xE0;
/// `asm:SongRequest` (then `asm:SampleRequest`, `asm:Tune1Request`, `asm:EffectRequest`).
pub const SONG_REQUEST: usize = 0x600;
/// `asm:SampleRequest`.
pub const SAMPLE_REQUEST: usize = 0x601;
/// `asm:Tune1Request`.
pub const TUNE1_REQUEST: usize = 0x602;
/// `asm:EffectRequest`.
pub const EFFECT_REQUEST: usize = 0x603;
/// `asm:Tune0Request`.
pub const TUNE0_REQUEST: usize = 0x604;
const REQUESTS: std::ops::Range<usize> = SONG_REQUEST..TUNE0_REQUEST + 1;
/// `asm:CurTune0`.
pub const CUR_TUNE0: usize = 0x605;
/// `asm:CurEffect`.
pub const CUR_EFFECT: usize = 0x606;
/// `asm:CurTune1`.
pub const CUR_TUNE1: usize = 0x607;
/// `asm:CurSample`.
pub const CUR_SAMPLE: usize = 0x608;
/// `asm:CurSong`.
pub const CUR_SONG: usize = 0x609;
/// `asm:BackgroundSample`.
pub const BACKGROUND_SAMPLE: usize = 0x5F6;
/// `asm:Tune0ScriptOffset`.
pub const TUNE0_SCRIPT_OFFSET: usize = 0x60E;
/// `Tune0Request` / `CurTune0` of the low-hearts beep (`asm:Audio_DriveTune0`).
pub const BEEP: u8 = 0x40;
/// The driver's own RAM: `asm:SongScriptPtrLo` .. `asm:NoteCounterTune1` and `asm:CurNoteLowPeriodTrg` ..
/// `asm:SongRepeatStartOffset`, the requests among them.
const ZERO_PAGE: std::ops::RangeInclusive<usize> = 0x66..=0x6F;
const PAGE: std::ops::RangeInclusive<usize> = 0x5F0..=0x61F;
/// The bytes of that state only tunes, noise effects and samples use: `asm:SampleCounter`,
/// `asm:SeaSfxCounter`, `asm:BackgroundSample`, `asm:CurTune0`, `asm:CurEffect`, `asm:CurTune1`,
/// `asm:CurSample`, `asm:Tune0ScriptOffset`, `asm:Tune1ScriptOffset`, `asm:UnusedSound61A`,
/// `asm:EffectScratch`, `asm:EffectCounter`, `asm:CustomEnvelopeOffsetTune1`, `asm:NoteLengthTune1`,
/// `asm:NoteCounterTune1`.  The music driver keeps them at zero.
pub const EFFECT_STATE: [usize; 15] =
    [0x5F2, 0x5F3, 0x5F6, 0x605, 0x606, 0x607, 0x608, 0x60E, 0x618, 0x61A, 0x68, 0x69, 0x6D, 0x6E, 0x6F];
/// The song's time keeping: `asm:CurSong`, `asm:SongPhraseIndex`, `asm:SongScriptPtrLo`/`Hi`, the four
/// tracks' note offsets, lengths and counters, the note length table, the noise loop point, the envelope
/// selector, the triangle's sustain flag and repeated passage.
pub const SONG_TIME: [usize; 22] = [
    0x609, 0x6C, 0x66, 0x67, 0x60A, 0x60B, 0x60C, 0x60D, 0x60F, 0x610, 0x611, 0x613, 0x615, 0x616, 0x617, 0x5F4,
    0x5F5, 0x619, 0x5F1, 0x61E, 0x61F, 0x5F0,
];
/// The register writes that end a square-0 tune (`asm:Audio_DriveTune0`, the script's end): constant volume
/// 0, length, period 0.
const END_TUNE0: [(u8, u8); 3] = [(0x00, 0x90), (0x03, 0x18), (0x02, 0x00)];
/// Most cycles a driver run may take (the real one takes about two thousand): more is a fault.
const RUN_LIMIT: u32 = 100_000;

/// Whether `addr` is a byte of the driver's own state.
#[must_use]
pub const fn driver_byte(addr: usize) -> bool {
    (addr >= *ZERO_PAGE.start() && addr <= *ZERO_PAGE.end()) || (addr >= *PAGE.start() && addr <= *PAGE.end())
}

/// Whether a song request (or `CurSong`) is one of the two jingles the effects carry: the end-of-level tune
/// (`$04`) or the item fanfare (`$08`).  The driver picks a song by value ($80 the title, $06 Zelda's song,
/// $01/$40/$10 the multi-phrase songs) and otherwise by its lowest set bit (`asm:Song_Drive`).
#[must_use]
pub const fn jingle(song: u8) -> bool {
    song & 0x80 == 0 && song != 0x06 && matches!(song & song.wrapping_neg(), 0x04 | 0x08)
}

/// Which parts of the sound are on (the switches).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mix {
    /// The songs.
    pub music: bool,
    /// Tunes, noise effects, samples and the two jingles.
    pub effects: bool,
    /// The low-hearts beep (an effect: heard only with `effects`).
    pub beep: bool,
}

impl Mix {
    /// Everything on: the machine's own sound.
    pub const ALL: Mix = Mix { music: true, effects: true, beep: true };
}

impl Default for Mix {
    fn default() -> Mix {
        Mix::ALL
    }
}

/// One of the sounds [`Tracks`] makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    /// The machine's own register writes: the game's sound as it always was.
    Game,
    /// The music driver's.
    Music,
    /// The effects driver's (the beep as [`Mix::beep`] says).
    Effects,
    /// A driver fed everything but the beep: the game without the beep.
    Quiet,
}

/// What the tracks saw (for checks and reports).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stats {
    /// Frames run with sound on.
    pub frames: u64,
    /// Of those, frames in which the machine's driver ran.
    pub driven: u64,
    /// Frames in which the mirror driver's register writes differed from the machine driver's.
    pub mismatched: u64,
    /// Driver runs that hit an unknown opcode or ran too long (their voice keeps its last state).
    pub faults: u64,
    /// Addresses a driver read or wrote outside its own state, `asm:PauseState`, the stack and the ROM.
    pub stray: Vec<u16>,
}

/// A register write by a driver, in CPU cycles from the start of its run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Write {
    /// Cycles from the start of the run.
    pub cycle: u32,
    /// `addr - $4000`.
    pub register: u8,
    /// The value written.
    pub value: u8,
}

/// The ROM's bank 0 (`$8000`) and fixed bank (`$C000`).
struct Banks<'a> {
    bank0: &'a [u8],
    fixed: &'a [u8],
}

impl<'a> Banks<'a> {
    fn of(rom: &'a Rom) -> Banks<'a> {
        let prg = rom.prg();
        Banks { bank0: &prg[..0x4000], fixed: &prg[prg.len() - 0x4000..] }
    }
}

/// One copy of the driver: its RAM (the machine's layout; only the driver's bytes, `asm:PauseState` and
/// the stack are used) and the register writes of its last run.
#[derive(Debug, Clone)]
struct Driver {
    ram: Box<[u8; 0x800]>,
    writes: Vec<Write>,
}

impl Driver {
    fn new() -> Driver {
        Driver { ram: Box::new([0; 0x800]), writes: Vec::with_capacity(64) }
    }

    /// Takes the machine's driver state (and pause state) as it stands.
    fn seed(&mut self, ram: &[u8; 0x800]) {
        for addr in ZERO_PAGE.chain(PAGE) {
            self.ram[addr] = ram[addr];
        }
        self.ram[PAUSE_STATE] = ram[PAUSE_STATE];
    }

    /// Runs the routine at `entry` from cycle `start` as if called; its register writes are added to
    /// `writes`.  Returns the cycle it returned at, or None on a fault.
    fn call(&mut self, banks: &Banks<'_>, entry: u16, start: u32, stray: &mut Vec<u16>) -> Option<u32> {
        let mut cpu = Cpu {
            ram: &mut self.ram,
            banks,
            writes: &mut self.writes,
            stray,
            a: 0,
            x: 0,
            y: 0,
            // A return address of $FFFF on the stack: the routine's RTS lands at $0000.
            s: 0xFD,
            p: 0x24,
            pc: entry,
            cycle: start,
        };
        cpu.ram[0x1FE] = 0xFF;
        cpu.ram[0x1FF] = 0xFF;
        while cpu.pc != 0 {
            if cpu.cycle - start > RUN_LIMIT || !cpu.step() {
                return None;
            }
        }
        Some(cpu.cycle)
    }
}

/// What a voice does with the requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Music,
    Effects,
    Quiet,
}

/// A driver with its synth and the actions due before its next run.
#[derive(Debug, Clone)]
struct Voice {
    kind: Kind,
    driver: Driver,
    synth: Synth,
    /// Made since the last frame: takes the machine's driver state and sound over first (`Tracks::join`).
    fresh: bool,
    /// Run `asm:Song_Silence` before the driver (the music: a jingle or a silence was requested).
    silence: bool,
    /// End the jingle before the driver (the effects: a song was requested over it).
    stop: bool,
    /// Writes due at the start of the next frame (a beep cut when the beep was switched off).
    due: Vec<(u8, u8)>,
    /// Its samples of the last frame.
    out: Vec<f32>,
}

impl Voice {
    fn new(kind: Kind, synth: Synth) -> Voice {
        Voice {
            kind,
            driver: Driver::new(),
            synth,
            fresh: true,
            silence: false,
            stop: false,
            due: Vec::new(),
            out: Vec::with_capacity(1024),
        }
    }

    /// Takes the machine's driver state, keeping what this voice plays.
    fn seed(&mut self, ram: &[u8; 0x800], beep: bool) {
        self.driver.seed(ram);
        let mine = &mut self.driver.ram;
        match self.kind {
            Kind::Music => {
                for addr in EFFECT_STATE {
                    mine[addr] = 0;
                }
                if jingle(mine[CUR_SONG]) {
                    mine[CUR_SONG] = 0;
                }
            }
            Kind::Effects => {
                if !jingle(mine[CUR_SONG]) {
                    mine[CUR_SONG] = 0;
                }
            }
            Kind::Quiet => {}
        }
        self.silence = false;
        self.stop = false;
        self.due.clear();
        if !beep {
            self.cut_beep();
        }
    }

    /// The voices of the machine's sound this voice takes over when it joins (`Synth::resume_only`).
    fn voices(&self, ram: &[u8; 0x800]) -> Voices {
        let song = ram[CUR_SONG];
        let (tune0, tune1, effect) = (ram[CUR_TUNE0] != 0, ram[CUR_TUNE1] != 0, ram[CUR_EFFECT] != 0);
        let mut voices = Voices::empty();
        match self.kind {
            Kind::Quiet => voices = Voices::all(),
            Kind::Music => {
                if song != 0 && !jingle(song) {
                    voices.set(Voices::SQUARE0, !tune0);
                    voices.set(Voices::SQUARE1, !tune1);
                    voices.insert(Voices::TRIANGLE);
                    voices.set(Voices::NOISE, !effect);
                }
            }
            Kind::Effects => {
                if jingle(song) {
                    voices |= Voices::SQUARE0 | Voices::SQUARE1 | Voices::TRIANGLE;
                }
                if tune0 {
                    voices.insert(Voices::SQUARE0);
                }
                if tune1 {
                    voices.insert(Voices::SQUARE1);
                }
                voices.set(Voices::NOISE, effect);
            }
        }
        voices
    }

    /// Ends a beep that is playing, as its script's end would (with the beep switched off).
    fn cut_beep(&mut self) {
        if self.kind != Kind::Music && self.driver.ram[CUR_TUNE0] == BEEP {
            self.driver.ram[CUR_TUNE0] = 0;
            self.due.extend_from_slice(&END_TUNE0);
        }
    }

    /// The machine's driver state changed by the game (`changed`) and its requests (`ram`), as this voice
    /// takes them.
    fn feed(&mut self, ram: &[u8; 0x800], changed: &[(usize, u8)], beep: bool) {
        let mine = &mut self.driver.ram;
        for &(addr, value) in changed {
            if REQUESTS.contains(&addr) || (self.kind == Kind::Music && EFFECT_STATE.contains(&addr)) {
                continue;
            }
            mine[addr] = value;
        }
        let mut requests = [0u8; 5];
        requests.copy_from_slice(&ram[REQUESTS]);
        let [song, _, tune1, _, tune0] = &mut requests;
        if !beep {
            *tune0 &= !BEEP;
        }
        match self.kind {
            Kind::Quiet => {}
            Kind::Effects => {
                if !jingle(*song) {
                    // A song's request replaces a jingle in the game; here it only stops it.
                    self.stop |= *song != 0 && jingle(mine[CUR_SONG]);
                    *song = 0;
                }
            }
            Kind::Music => {
                // A jingle replaces the song in the game, and Tune1Request bit 7 silences it before the death
                // tune (`asm:Audio_DriveTune1`): the music stops the same way, before its driver runs.
                self.silence |= jingle(*song) || *tune1 & 0x80 != 0;
                if jingle(*song) {
                    *song = 0;
                }
                requests[1] = 0;
                requests[2] = 0;
                // `EffectRequest` $80 (`asm:Sample_Silence`) only stops samples and tune 1, which the music never
                // has; it runs anyway, so that the music's writes in the frame the game silences everything come
                // at the game's cycles.
                requests[3] &= 0x80;
                requests[4] &= 0x80;
            }
        }
        mine[REQUESTS].copy_from_slice(&requests);
    }

    /// Runs the driver for a frame in which the machine's ran.  Its writes are in `driver.writes`; returns
    /// the cycle (from the run's start) of `asm:Audio_Drive`'s first write, or None on a fault.
    fn drive(&mut self, banks: &Banks<'_>, stray: &mut Vec<u16>) -> Option<u32> {
        self.driver.writes.clear();
        let mut cycle = 0;
        if std::mem::take(&mut self.silence) {
            cycle = self.driver.call(banks, SONG_SILENCE, cycle, stray)?;
        }
        if std::mem::take(&mut self.stop) {
            // The jingle's voices go quiet as a tune's square does at its end and a rest does the triangle's.
            let ram = &mut self.driver.ram;
            ram[CUR_SONG] = 0;
            let writes = &mut self.driver.writes;
            if ram[CUR_TUNE0] == 0 {
                writes.push(Write { cycle, register: 0x00, value: 0x90 });
            }
            if ram[CUR_TUNE1] == 0 {
                writes.push(Write { cycle, register: 0x04, value: 0x90 });
            }
            writes.push(Write { cycle, register: 0x08, value: 0x00 });
        }
        let first = self.driver.writes.len();
        self.driver.call(banks, AUDIO_DRIVE, cycle, stray)?;
        self.driver.writes.get(first).map(|write| write.cycle)
    }
}

/// The game's sound as music and sound effects (module docs).  Follows one machine: [`Tracks::prepare`]
/// before each frame it runs, [`Tracks::run`] after.
#[derive(Debug, Clone)]
pub struct Tracks {
    rom: Arc<Rom>,
    mix: Mix,
    split: bool,
    warm: bool,
    game: Synth,
    game_out: Vec<f32>,
    mirror: Driver,
    music: Option<Voice>,
    effects: Option<Voice>,
    quiet: Option<Voice>,
    pause: u8,
    changed: Vec<(usize, u8)>,
    outside: Vec<Event>,
    samples: Vec<Event>,
    events: Vec<Event>,
    stats: Stats,
}

impl Tracks {
    /// Sound for `nes` from now on, at `rate` samples per second, as `mix` says; `split`: two tracks
    /// (music, effects) instead of one ([`Tracks::run`]); `warm`: keep both tracks running whatever the mix,
    /// so that [`Tracks::set_mix`] takes effect at once and seamlessly (a player that switches while it
    /// plays).  Takes the machine's sound over as [`Synth::resume`] does.
    #[must_use]
    pub fn new(nes: &ZCore, rate: f64, mix: Mix, split: bool, warm: bool) -> Tracks {
        let mut tracks = Tracks {
            rom: nes.rom().clone(),
            mix,
            split,
            warm,
            game: Synth::new(rate),
            game_out: Vec::with_capacity(1024),
            mirror: Driver::new(),
            music: None,
            effects: None,
            quiet: None,
            pause: 0,
            changed: Vec::with_capacity(64),
            outside: Vec::new(),
            samples: Vec::new(),
            events: Vec::with_capacity(128),
            stats: Stats::default(),
        };
        tracks.resume(nes);
        tracks
    }

    /// The switches.
    #[must_use]
    pub const fn mix(&self) -> Mix {
        self.mix
    }

    /// Whether [`Tracks::run`] makes two tracks.
    #[must_use]
    pub const fn split(&self) -> bool {
        self.split
    }

    /// What the tracks saw so far.
    #[must_use]
    pub const fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The driver RAM of the mirror (`None`) or of a track's driver, if it runs.
    #[must_use]
    pub fn driver_ram(&self, track: Option<Track>) -> Option<&[u8; 0x800]> {
        match track {
            None => Some(&self.mirror.ram),
            Some(track) => self.voice(track).map(|voice| &*voice.driver.ram),
        }
    }

    /// The register writes of the last frame's run of the mirror (`None`) or of a track's driver.
    #[must_use]
    pub fn driver_writes(&self, track: Option<Track>) -> &[Write] {
        match track {
            None => &self.mirror.writes,
            Some(track) => self.voice(track).map_or(&[], |voice| &voice.driver.writes),
        }
    }

    /// A track's samples of the last frame (empty for one that does not run).
    #[must_use]
    pub fn samples(&self, track: Track) -> &[f32] {
        match track {
            Track::Game => &self.game_out,
            _ => self.voice(track).map_or(&[], |voice| &voice.out),
        }
    }

    fn voice(&self, track: Track) -> Option<&Voice> {
        match track {
            Track::Game => None,
            Track::Music => self.music.as_ref(),
            Track::Effects => self.effects.as_ref(),
            Track::Quiet => self.quiet.as_ref(),
        }
    }

    /// Changes the output rate of every track (`Synth::set_rate`).
    pub fn set_rate(&mut self, rate: f64) {
        self.game.set_rate(rate);
        for voice in [&mut self.music, &mut self.effects, &mut self.quiet].into_iter().flatten() {
            voice.synth.set_rate(rate);
        }
    }

    /// Switches music, effects and the beep from the next frame on.  A beep that is playing ends at once.
    pub fn set_mix(&mut self, mix: Mix) {
        let was = self.mix;
        self.mix = mix;
        if was.beep && !mix.beep {
            if let Some(voice) = &mut self.effects {
                voice.cut_beep();
            }
        }
        self.arrange(was.beep);
    }

    /// Makes the voices the mix needs (and drops the quiet one when the beep is on).
    fn arrange(&mut self, renew_quiet: bool) {
        let both = self.warm || self.split || self.mix != Mix::ALL;
        for (slot, kind) in [(&mut self.music, Kind::Music), (&mut self.effects, Kind::Effects)] {
            if both && slot.is_none() {
                // Made from the game's synth, so its output keeps the same sample clock; it takes its voices
                // over before the next frame (`join`).
                *slot = Some(Voice::new(kind, self.game.clone()));
            }
        }
        let quiet = self.mix.music && self.mix.effects && !self.mix.beep && !self.split;
        if !quiet {
            self.quiet = None;
        } else if self.quiet.is_none() || renew_quiet {
            // The game's sound up to now, from now on without the beep.
            let mut voice = Voice::new(Kind::Quiet, self.game.clone());
            voice.driver = self.mirror.clone();
            voice.fresh = false;
            voice.cut_beep();
            self.quiet = Some(voice);
        }
    }

    /// Takes the machine's sound over as it stands (a start, a seek, a cell that starts being heard): the
    /// game's synth resumes ([`Synth::resume`]), every driver takes the machine's driver state, and each
    /// track's synth the voices it plays at that moment.
    pub fn resume(&mut self, nes: &ZCore) {
        let ram = nes.ram();
        self.game.resume(nes);
        self.mirror.seed(ram);
        self.quiet = None;
        self.music = None;
        self.effects = None;
        self.arrange(true);
        self.join(nes);
    }

    /// The voices made since the last frame take the machine's driver state and sound over.
    fn join(&mut self, nes: &ZCore) {
        let ram = nes.ram();
        let beep = self.mix.beep;
        for voice in [&mut self.music, &mut self.effects].into_iter().flatten() {
            if std::mem::take(&mut voice.fresh) {
                voice.seed(ram, beep);
                let voices = voice.voices(ram);
                voice.synth.resume_only(nes, voices);
            }
        }
    }

    /// Call before each frame `nes` runs: what the game wrote to the driver's state since the last frame
    /// goes to every driver, each taking its own requests.
    pub fn prepare(&mut self, nes: &ZCore) {
        self.join(nes);
        let ram = nes.ram();
        self.changed.clear();
        for addr in ZERO_PAGE.chain(PAGE).chain([PAUSE_STATE]) {
            if ram[addr] != self.mirror.ram[addr] {
                self.changed.push((addr, ram[addr]));
                self.mirror.ram[addr] = ram[addr];
            }
        }
        self.pause = ram[PAUSE_STATE];
        let beep = self.mix.beep;
        for voice in [&mut self.music, &mut self.effects, &mut self.quiet].into_iter().flatten() {
            voice.feed(ram, &self.changed, beep);
        }
    }

    /// Call after each frame `nes` ran (after [`Tracks::prepare`]): appends the frame's samples to `out`,
    /// mono as the mix says, or with `split` (music, effects) pairs.
    pub fn run(&mut self, nes: &ZCore, out: &mut Vec<f32>) {
        let Some(frame) = nes.sound() else { return };
        self.stats.frames += 1;
        self.game_out.clear();
        self.game.run(frame, &mut self.game_out);
        if self.music.is_some() || self.effects.is_some() || self.quiet.is_some() {
            self.drive(frame);
        }
        // Every synth makes the same count (one sample clock); a voice that does not run is silence.
        fn samples(voice: Option<&Voice>, n: usize) -> &[f32] {
            voice.map_or(&[], |voice| &voice.out[..voice.out.len().min(n)])
        }
        let n = self.game_out.len();
        if self.split {
            let music = samples(self.music.as_ref().filter(|_| self.mix.music), n);
            let effects = samples(self.effects.as_ref().filter(|_| self.mix.effects), n);
            out.extend((0..n).flat_map(|i| {
                [music.get(i).copied().unwrap_or(0.0), effects.get(i).copied().unwrap_or(0.0)]
            }));
            return;
        }
        let mono = match (self.mix.music, self.mix.effects, self.mix.beep) {
            (true, true, true) => &self.game_out[..],
            (true, true, false) => samples(self.quiet.as_ref(), n),
            (true, false, _) => samples(self.music.as_ref(), n),
            (false, true, _) => samples(self.effects.as_ref(), n),
            (false, false, _) => &[],
        };
        out.extend_from_slice(mono);
        out.resize(out.len() + n - mono.len(), 0.0);
    }

    /// Runs the drivers if the machine's ran this frame, and every voice's synth.
    fn drive(&mut self, frame: Frame<'_>) {
        let rom = self.rom.clone();
        let banks = Banks::of(&rom);
        self.samples.clear();
        self.outside.clear();
        let mut writes = frame.events().iter().filter(|event| event.register != SAMPLE);
        self.samples.extend(frame.events().iter().filter(|event| event.register == SAMPLE));
        // The driver's first write: the frame counter's sync when the game runs, the channels' cut while it is
        // paused (`asm:Audio_Drive`).  Nothing else writes either first in a frame.
        let marker = if self.pause == 0 { (0x17, 0xFF) } else { (0x15, 0x00) };
        let first = frame.events().iter().find(|event| event.register != SAMPLE);
        let at = first.filter(|event| (event.register, event.value) == marker).map(|event| event.cycle);
        let mut ran = false;
        self.mirror.writes.clear();
        if let Some(at) = at {
            self.stats.driven += 1;
            if self.mirror.call(&banks, AUDIO_DRIVE, 0, &mut self.stats.stray).is_some() {
                ran = true;
            } else {
                self.stats.faults += 1;
            }
            // The machine's driver wrote what the mirror wrote; the writes after those are the game's own.
            let mut matched = true;
            for write in &self.mirror.writes {
                match writes.next() {
                    Some(event) if (event.register, event.value) == (write.register, write.value) => {}
                    _ => matched = false,
                }
            }
            if !matched {
                self.stats.mismatched += 1;
            }
            self.outside.extend(writes);
            if ran {
                for voice in [&mut self.music, &mut self.effects, &mut self.quiet].into_iter().flatten() {
                    voice.out.clear();
                    let mark = voice.drive(&banks, &mut self.stats.stray);
                    if mark.is_none() {
                        self.stats.faults += 1;
                        voice.driver.writes.clear();
                    }
                    // Each run starts as many cycles before its first write as the machine's driver did.
                    let start = at.saturating_sub(mark.unwrap_or(0));
                    Self::play(voice, frame.cycles(), &self.outside, &self.samples, &mut self.events, start);
                }
            }
        } else {
            self.outside.extend(writes);
        }
        if !ran {
            for voice in [&mut self.music, &mut self.effects, &mut self.quiet].into_iter().flatten() {
                voice.out.clear();
                voice.driver.writes.clear();
                Self::play(voice, frame.cycles(), &self.outside, &self.samples, &mut self.events, 0);
            }
        }
        // The music plays no samples, but the console mixes the triangle and the noise with the DMC's resting
        // level: between samples the music's DMC rests where the game's does.
        if let (Some(voice), Some(level)) = (&mut self.music, self.game.resting_dmc()) {
            if self.mirror.ram[CUR_SAMPLE] == 0 {
                voice.synth.rest_dmc(level);
            }
        }
    }

    /// One frame of a voice's synth: its driver's writes (its run starting at cycle `start` of the frame), the
    /// writes due, the game's own writes and, for the voices that play samples, the DMC's sample bytes.
    fn play(voice: &mut Voice, cycles: u32, outside: &[Event], samples: &[Event], events: &mut Vec<Event>, start: u32) {
        events.clear();
        events.extend(voice.due.drain(..).map(|(register, value)| Event { cycle: 0, register, value }));
        events.extend(voice.driver.writes.iter().map(|write| Event {
            cycle: start + write.cycle,
            register: write.register,
            value: write.value,
        }));
        events.extend_from_slice(outside);
        if voice.kind != Kind::Music {
            events.extend_from_slice(samples);
        }
        events.sort_by_key(|event| event.cycle);
        voice.synth.run(Frame { events, cycles }, &mut voice.out);
    }
}

/// The 6502 a driver runs on: the driver's RAM (mirrored to `$1FFF`), bank 0 at `$8000`, the fixed bank at
/// `$C000`; writes to `$4000-$4017` are noted with their cycle.  No interrupts; no decimal mode (the
/// console's CPU has none); an unofficial opcode is a fault.
struct Cpu<'a, 'b> {
    ram: &'a mut [u8; 0x800],
    banks: &'a Banks<'b>,
    writes: &'a mut Vec<Write>,
    stray: &'a mut Vec<u16>,
    a: u8,
    x: u8,
    y: u8,
    s: u8,
    p: u8,
    pc: u16,
    cycle: u32,
}

const C: u8 = 0x01;
const Z: u8 = 0x02;
const V: u8 = 0x40;
const N: u8 = 0x80;

/// Base cycles of each official opcode (0: unofficial).
#[rustfmt::skip]
const CYCLES: [u8; 256] = [
    7,6,0,0,0,3,5,0,3,2,2,0,0,4,6,0, 2,5,0,0,0,4,6,0,2,4,0,0,0,4,7,0,
    6,6,0,0,3,3,5,0,4,2,2,0,4,4,6,0, 2,5,0,0,0,4,6,0,2,4,0,0,0,4,7,0,
    6,6,0,0,0,3,5,0,3,2,2,0,3,4,6,0, 2,5,0,0,0,4,6,0,2,4,0,0,0,4,7,0,
    6,6,0,0,0,3,5,0,4,2,2,0,5,4,6,0, 2,5,0,0,0,4,6,0,2,4,0,0,0,4,7,0,
    0,6,0,0,3,3,3,0,2,0,2,0,4,4,4,0, 2,6,0,0,4,4,4,0,2,5,2,0,0,5,0,0,
    2,6,2,0,3,3,3,0,2,2,2,0,4,4,4,0, 2,5,0,0,4,4,4,0,2,4,2,0,4,4,4,0,
    2,6,0,0,3,3,5,0,2,2,2,0,4,4,6,0, 2,5,0,0,0,4,6,0,2,4,0,0,0,4,7,0,
    2,6,0,0,3,3,5,0,2,2,2,0,4,4,6,0, 2,5,0,0,0,4,6,0,2,4,0,0,0,4,7,0,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Accumulator,
    Immediate,
    ZeroPage,
    ZeroPageX,
    ZeroPageY,
    Absolute,
    AbsoluteX,
    AbsoluteY,
    IndirectX,
    IndirectY,
}

impl Cpu<'_, '_> {
    fn read(&mut self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x1FFF => {
                let at = usize::from(addr & 0x7FF);
                if !(driver_byte(at) || at == PAUSE_STATE || (0x100..0x200).contains(&at)) {
                    self.note_stray(addr);
                }
                self.ram[at]
            }
            0x8000..=0xBFFF => self.banks.bank0[usize::from(addr - 0x8000)],
            0xC000..=0xFFFF => self.banks.fixed[usize::from(addr - 0xC000)],
            _ => {
                self.note_stray(addr);
                0
            }
        }
    }

    /// A write, on the last cycle of an instruction that ends at `end`.
    fn write(&mut self, addr: u16, value: u8, end: u32) {
        match addr {
            0x0000..=0x1FFF => {
                let at = usize::from(addr & 0x7FF);
                if !(driver_byte(at) || (0x100..0x200).contains(&at)) {
                    self.note_stray(addr);
                }
                self.ram[at] = value;
            }
            0x4000..=0x4013 | 0x4015 | 0x4017 => {
                self.writes.push(Write { cycle: end - 1, register: (addr - 0x4000) as u8, value });
            }
            _ => self.note_stray(addr),
        }
    }

    fn note_stray(&mut self, addr: u16) {
        if !self.stray.contains(&addr) {
            self.stray.push(addr);
        }
    }

    fn fetch(&mut self) -> u8 {
        let byte = self.read(self.pc);
        self.pc = self.pc.wrapping_add(1);
        byte
    }

    fn fetch_word(&mut self) -> u16 {
        let lo = self.fetch();
        u16::from_le_bytes([lo, self.fetch()])
    }

    fn push(&mut self, value: u8) {
        self.ram[0x100 + usize::from(self.s)] = value;
        self.s = self.s.wrapping_sub(1);
    }

    fn pull(&mut self) -> u8 {
        self.s = self.s.wrapping_add(1);
        self.ram[0x100 + usize::from(self.s)]
    }

    fn set(&mut self, flag: u8, on: bool) {
        if on {
            self.p |= flag;
        } else {
            self.p &= !flag;
        }
    }

    fn nz(&mut self, value: u8) -> u8 {
        self.set(Z, value == 0);
        self.set(N, value & 0x80 != 0);
        value
    }

    fn compare(&mut self, register: u8, value: u8) {
        self.set(C, register >= value);
        self.nz(register.wrapping_sub(value));
    }

    fn add(&mut self, value: u8) {
        let sum = u16::from(self.a) + u16::from(value) + u16::from(self.p & C);
        let result = sum as u8;
        self.set(C, sum > 0xFF);
        self.set(V, (!(self.a ^ value) & (self.a ^ result) & 0x80) != 0);
        self.a = self.nz(result);
    }

    /// The operand's address for `mode` and whether indexing crossed a page.
    fn address(&mut self, mode: Mode) -> (u16, bool) {
        let cross = |base: u16, addr: u16| base & 0xFF00 != addr & 0xFF00;
        match mode {
            Mode::ZeroPage => (u16::from(self.fetch()), false),
            Mode::ZeroPageX => (u16::from(self.fetch().wrapping_add(self.x)), false),
            Mode::ZeroPageY => (u16::from(self.fetch().wrapping_add(self.y)), false),
            Mode::Absolute => (self.fetch_word(), false),
            Mode::AbsoluteX | Mode::AbsoluteY => {
                let base = self.fetch_word();
                let index = if mode == Mode::AbsoluteX { self.x } else { self.y };
                let addr = base.wrapping_add(u16::from(index));
                (addr, cross(base, addr))
            }
            Mode::IndirectX => {
                let at = self.fetch().wrapping_add(self.x);
                let lo = self.read(u16::from(at));
                (u16::from_le_bytes([lo, self.read(u16::from(at.wrapping_add(1)))]), false)
            }
            Mode::IndirectY => {
                let at = self.fetch();
                let lo = self.read(u16::from(at));
                let base = u16::from_le_bytes([lo, self.read(u16::from(at.wrapping_add(1)))]);
                let addr = base.wrapping_add(u16::from(self.y));
                (addr, cross(base, addr))
            }
            Mode::Accumulator | Mode::Immediate => (0, false),
        }
    }

    /// The addressing mode of an opcode of the regular groups (`aaabbbcc`), or None for the others.
    fn mode(op: u8) -> Option<Mode> {
        let bbb = (op >> 2) & 7;
        Some(match op & 3 {
            1 => [
                Mode::IndirectX,
                Mode::ZeroPage,
                Mode::Immediate,
                Mode::Absolute,
                Mode::IndirectY,
                Mode::ZeroPageX,
                Mode::AbsoluteY,
                Mode::AbsoluteX,
            ][usize::from(bbb)],
            2 => match bbb {
                0 => Mode::Immediate,
                1 => Mode::ZeroPage,
                2 if op < 0x80 => Mode::Accumulator,
                3 => Mode::Absolute,
                5 if matches!(op, 0x96 | 0xB6) => Mode::ZeroPageY,
                5 => Mode::ZeroPageX,
                7 if op == 0xBE => Mode::AbsoluteY,
                7 => Mode::AbsoluteX,
                _ => return None,
            },
            _ => match bbb {
                0 if op >= 0xA0 => Mode::Immediate,
                1 => Mode::ZeroPage,
                3 if op != 0x4C && op != 0x6C => Mode::Absolute,
                5 => Mode::ZeroPageX,
                7 => Mode::AbsoluteX,
                _ => return None,
            },
        })
    }

    /// Runs one instruction.  False on an opcode the driver should never run.
    fn step(&mut self) -> bool {
        let start = self.cycle;
        let op = self.fetch();
        let base = CYCLES[usize::from(op)];
        if base == 0 || op == 0x00 || op == 0x40 {
            return false;
        }
        let mut cycles = u32::from(base);
        let Some(mode) = Self::mode(op).filter(|_| !matches!(op, 0x20 | 0x60 | 0x4C | 0x6C) && op & 0x1F != 0x10)
        else {
            self.special(op, &mut cycles);
            self.cycle = start + cycles;
            return true;
        };
        let (addr, crossed) = self.address(mode);
        // Reads through an index that crosses a page take a cycle more; stores and read-modify-writes always
        // take it (counted in CYCLES).
        let reads = op >> 5 != 4 && (op & 3 == 1 || matches!(op, 0xBE | 0xBC));
        if crossed && reads {
            cycles += 1;
        }
        let end = start + cycles;
        let operand = |cpu: &mut Self| if mode == Mode::Immediate { cpu.fetch() } else { cpu.read(addr) };
        match (op & 3, op >> 5) {
            (1, 0) => {
                let v = operand(self);
                self.a = self.nz(self.a | v);
            }
            (1, 1) => {
                let v = operand(self);
                self.a = self.nz(self.a & v);
            }
            (1, 2) => {
                let v = operand(self);
                self.a = self.nz(self.a ^ v);
            }
            (1, 3) => {
                let v = operand(self);
                self.add(v);
            }
            (1, 4) => self.write(addr, self.a, end),
            (1, 5) => {
                let v = operand(self);
                self.a = self.nz(v);
            }
            (1, 6) => {
                let v = operand(self);
                self.compare(self.a, v);
            }
            (1, 7) => {
                let v = operand(self);
                self.add(!v);
            }
            (2, 4) => self.write(addr, self.x, end),
            (2, 5) => {
                let v = operand(self);
                self.x = self.nz(v);
            }
            (2, aaa) => {
                // ASL ROL LSR ROR DEC INC, on the accumulator or memory.
                let value = if mode == Mode::Accumulator { self.a } else { self.read(addr) };
                let carry = self.p & C;
                let result = match aaa {
                    0 => {
                        self.set(C, value & 0x80 != 0);
                        value << 1
                    }
                    1 => {
                        self.set(C, value & 0x80 != 0);
                        (value << 1) | carry
                    }
                    2 => {
                        self.set(C, value & 1 != 0);
                        value >> 1
                    }
                    3 => {
                        self.set(C, value & 1 != 0);
                        (value >> 1) | (carry << 7)
                    }
                    6 => value.wrapping_sub(1),
                    _ => value.wrapping_add(1),
                };
                self.nz(result);
                if mode == Mode::Accumulator {
                    self.a = result;
                } else {
                    self.write(addr, result, end);
                }
            }
            (0, 1) => {
                let v = self.read(addr);
                self.set(Z, self.a & v == 0);
                self.set(V, v & 0x40 != 0);
                self.set(N, v & 0x80 != 0);
            }
            (0, 4) => self.write(addr, self.y, end),
            (0, 5) => {
                let v = operand(self);
                self.y = self.nz(v);
            }
            (0, 6) => {
                let v = operand(self);
                self.compare(self.y, v);
            }
            (0, 7) => {
                let v = operand(self);
                self.compare(self.x, v);
            }
            _ => return false,
        }
        self.cycle = start + cycles;
        true
    }

    /// Branches, jumps, calls, returns and the one-byte instructions.
    fn special(&mut self, op: u8, cycles: &mut u32) {
        if op & 0x1F == 0x10 {
            let offset = self.fetch() as i8;
            let flag = [N, V, C, Z][usize::from(op >> 6)];
            if (self.p & flag != 0) == (op & 0x20 != 0) {
                let target = self.pc.wrapping_add_signed(i16::from(offset));
                *cycles += if target & 0xFF00 == self.pc & 0xFF00 { 1 } else { 2 };
                self.pc = target;
            }
            return;
        }
        match op {
            0x20 => {
                let target = self.fetch_word();
                let back = self.pc.wrapping_sub(1);
                self.push((back >> 8) as u8);
                self.push(back as u8);
                self.pc = target;
            }
            0x60 => {
                let lo = self.pull();
                self.pc = u16::from_le_bytes([lo, self.pull()]).wrapping_add(1);
            }
            0x4C => self.pc = self.fetch_word(),
            0x6C => {
                let at = self.fetch_word();
                let lo = self.read(at);
                // The 6502 does not carry into the high byte of the pointer.
                let hi = self.read((at & 0xFF00) | (at.wrapping_add(1) & 0xFF));
                self.pc = u16::from_le_bytes([lo, hi]);
            }
            0x08 => self.push(self.p | 0x30),
            0x28 => self.p = (self.pull() & !0x10) | 0x20,
            0x48 => self.push(self.a),
            0x68 => {
                let v = self.pull();
                self.a = self.nz(v);
            }
            0x18 => self.set(C, false),
            0x38 => self.set(C, true),
            0x58 => self.set(0x04, false),
            0x78 => self.set(0x04, true),
            0xB8 => self.set(V, false),
            0xD8 => self.set(0x08, false),
            0xF8 => self.set(0x08, true),
            0x88 => self.y = self.nz(self.y.wrapping_sub(1)),
            0xC8 => self.y = self.nz(self.y.wrapping_add(1)),
            0xCA => self.x = self.nz(self.x.wrapping_sub(1)),
            0xE8 => self.x = self.nz(self.x.wrapping_add(1)),
            0xA8 => self.y = self.nz(self.a),
            0x98 => self.a = self.nz(self.y),
            0xAA => self.x = self.nz(self.a),
            0x8A => self.a = self.nz(self.x),
            0xBA => self.x = self.nz(self.s),
            0x9A => self.s = self.x,
            _ => {} // NOP
        }
    }
}
