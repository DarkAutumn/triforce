//! Sound: what the console's speaker would play, computed beside the machine rather than inside it.
//!
//! The emulated APU ([`crate::apu`]) keeps only what the CPU can observe, and [`MachineState`] has no sound
//! state: a state file or a replay is the same with or without sound.  Sound is made from what the program
//! writes instead.  The bus notes every write to `$4000-$4017` and every sample byte the DMC fetches (the
//! [`Tap`]: the last value of each register always, the timed writes of the last frame while
//! [`ZCore::set_sound`] is on), and a [`Synth`] replays them through its own pulse, triangle, noise and
//! DMC channels, the mixer and the output filters ([`Synth::run`], one frame at a time).  Nothing the synth
//! computes is read back by the machine.
//!
//! A synth that starts in the middle of a run ([`Synth::resume`]) takes the registers the tap has seen and
//! the machine's length counters, so held notes go on.  After [`ZCore::restore`] the registers are unknown
//! (a state file does not hold them): each channel is silent until the program writes it again, which a
//! music driver does on its next note.
//!
//! See <https://www.nesdev.org/wiki/APU> (channels, frame counter) and `APU_Mixer` (the lookup mixer, the
//! 90 Hz and 440 Hz high-pass and 14 kHz low-pass output filters).
//!
//! [`MachineState`]: crate::MachineState
//! [`ZCore::set_sound`]: crate::ZCore::set_sound
//! [`ZCore::restore`]: crate::ZCore::restore

use std::collections::VecDeque;

use crate::ZCore;

/// CPU cycles per second of an NTSC console (the 21.477272 MHz master clock divided by 12).
pub const CPU_HZ: f64 = 39_375_000.0 / 22.0;

/// `Event::register` of a sample byte the DMC fetched.
const SAMPLE: u8 = 0xFF;
/// Registers `$4000-$4017`.
const REGISTERS: usize = 0x18;

/// A register write or a fetched DMC sample byte, timed in CPU cycles from the start of its frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    cycle: u32,
    /// `addr - $4000`, or [`SAMPLE`].
    register: u8,
    value: u8,
}

/// The bus's notes for sound (module docs).  Never part of the machine state.
#[derive(Debug, Default)]
pub(crate) struct Tap {
    registers: [u8; REGISTERS],
    /// Bit `r`: `registers[r]` was written since power on or the last restore.
    known: u32,
    /// The timed events of the frame being run, while sound is on.
    log: Option<Vec<Event>>,
    /// The APU's CPU cycle counter at the start of the frame.
    start: u32,
    /// CPU cycles of the last frame.
    cycles: u32,
}

impl Tap {
    #[inline]
    pub(crate) fn write(&mut self, cycle: u32, addr: u16, value: u8) {
        let register = (addr - 0x4000) as u8;
        self.registers[usize::from(register)] = value;
        self.known |= 1 << register;
        if let Some(log) = &mut self.log {
            log.push(Event { cycle: cycle.wrapping_sub(self.start), register, value });
        }
    }

    #[inline]
    pub(crate) fn sample(&mut self, cycle: u32, value: u8) {
        if let Some(log) = &mut self.log {
            log.push(Event { cycle: cycle.wrapping_sub(self.start), register: SAMPLE, value });
        }
    }

    pub(crate) fn set_logging(&mut self, on: bool) {
        match (on, &self.log) {
            (true, None) => self.log = Some(Vec::with_capacity(64)),
            (false, Some(_)) => self.log = None,
            _ => {}
        }
        self.cycles = 0;
    }

    pub(crate) fn begin_frame(&mut self, cycle: u32) {
        self.start = cycle;
        if let Some(log) = &mut self.log {
            log.clear();
        }
    }

    pub(crate) fn end_frame(&mut self, cycle: u32) {
        self.cycles = cycle.wrapping_sub(self.start);
    }

    /// The machine went somewhere else: what was written before says nothing about it.
    pub(crate) fn forget(&mut self) {
        self.known = 0;
        self.cycles = 0;
        if let Some(log) = &mut self.log {
            log.clear();
        }
    }

    pub(crate) fn frame(&self) -> Option<Frame<'_>> {
        self.log.as_deref().map(|events| Frame { events, cycles: self.cycles })
    }

    fn register(&self, register: usize) -> Option<u8> {
        (self.known & (1 << register) != 0).then(|| self.registers[register])
    }
}

/// The sound events of one frame ([`ZCore::sound`]).
///
/// [`ZCore::sound`]: crate::ZCore::sound
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    events: &'a [Event],
    cycles: u32,
}

impl Frame<'_> {
    /// CPU cycles the frame ran.
    #[must_use]
    pub const fn cycles(&self) -> u32 {
        self.cycles
    }
}

const LENGTHS: [u8; 32] = [
    10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14, 12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72,
    26, 16, 28, 32, 30,
];
const DUTIES: [[u8; 8]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0],
];
const TRIANGLE: [u8; 32] = [
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];
/// Noise and DMC timer periods, in CPU cycles (NTSC).
const NOISE_PERIODS: [u16; 16] = [4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068];
const DMC_PERIODS: [u16; 16] = [428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54];
/// The frame sequencer's quarter-frame cycles; the second and the last also clock a half frame.
const QUARTERS: [u32; 3] = [7457, 14913, 22371];
const LAST_STEP: [u32; 2] = [29829, 37281];

/// The volume envelope of a pulse or the noise channel.
#[derive(Debug, Default, Clone, Copy)]
struct Envelope {
    start: bool,
    /// Loop (and halt the length counter).
    halt: bool,
    constant: bool,
    volume: u8,
    divider: u8,
    decay: u8,
}

impl Envelope {
    fn write(&mut self, value: u8) {
        self.halt = value & 0x20 != 0;
        self.constant = value & 0x10 != 0;
        self.volume = value & 0x0F;
    }

    fn clock(&mut self) {
        if self.start {
            self.start = false;
            self.decay = 15;
            self.divider = self.volume;
        } else if self.divider > 0 {
            self.divider -= 1;
        } else {
            self.divider = self.volume;
            if self.decay > 0 {
                self.decay -= 1;
            } else if self.halt {
                self.decay = 15;
            }
        }
    }

    const fn level(&self) -> u8 {
        if self.constant { self.volume } else { self.decay }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Pulse {
    /// The second pulse channel (its sweep subtracts one less).
    second: bool,
    enabled: bool,
    envelope: Envelope,
    duty: u8,
    position: u8,
    period: u16,
    timer: u16,
    length: u8,
    sweep_on: bool,
    sweep_period: u8,
    sweep_negate: bool,
    sweep_shift: u8,
    sweep_reload: bool,
    sweep_divider: u8,
}

impl Pulse {
    fn write(&mut self, register: u8, value: u8) {
        match register {
            0 => {
                self.duty = value >> 6;
                self.envelope.write(value);
            }
            1 => {
                self.sweep_on = value & 0x80 != 0;
                self.sweep_period = (value >> 4) & 7;
                self.sweep_negate = value & 0x08 != 0;
                self.sweep_shift = value & 7;
                self.sweep_reload = true;
            }
            2 => self.period = (self.period & 0x700) | u16::from(value),
            _ => {
                self.period = (self.period & 0xFF) | (u16::from(value & 7) << 8);
                if self.enabled {
                    self.length = LENGTHS[usize::from(value >> 3)];
                }
                self.position = 0;
                self.envelope.start = true;
            }
        }
    }

    /// The period the sweep unit would set next.
    fn target(&self) -> u16 {
        let change = self.period >> self.sweep_shift;
        if !self.sweep_negate {
            self.period + change
        } else if self.second {
            self.period.saturating_sub(change)
        } else {
            self.period.saturating_sub(change + 1)
        }
    }

    fn muted(&self) -> bool {
        self.period < 8 || self.target() > 0x7FF
    }

    /// One APU cycle (two CPU cycles).
    fn clock(&mut self) {
        if self.timer == 0 {
            self.timer = self.period;
            self.position = (self.position + 1) & 7;
        } else {
            self.timer -= 1;
        }
    }

    fn half_frame(&mut self) {
        if self.length > 0 && !self.envelope.halt {
            self.length -= 1;
        }
        if self.sweep_divider == 0 && self.sweep_on && self.sweep_shift > 0 && !self.muted() {
            self.period = self.target();
        }
        if self.sweep_divider == 0 || self.sweep_reload {
            self.sweep_divider = self.sweep_period;
            self.sweep_reload = false;
        } else {
            self.sweep_divider -= 1;
        }
    }

    fn output(&self) -> u8 {
        if self.length == 0 || DUTIES[usize::from(self.duty)][usize::from(self.position)] == 0 || self.muted() {
            0
        } else {
            self.envelope.level()
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Triangle {
    enabled: bool,
    /// Halts the length counter and keeps the linear counter reloading.
    control: bool,
    reload_value: u8,
    reload: bool,
    linear: u8,
    period: u16,
    timer: u16,
    length: u8,
    position: u8,
}

impl Triangle {
    fn write(&mut self, register: u8, value: u8) {
        match register {
            0 => {
                self.control = value & 0x80 != 0;
                self.reload_value = value & 0x7F;
            }
            2 => self.period = (self.period & 0x700) | u16::from(value),
            3 => {
                self.period = (self.period & 0xFF) | (u16::from(value & 7) << 8);
                if self.enabled {
                    self.length = LENGTHS[usize::from(value >> 3)];
                }
                self.reload = true;
            }
            _ => {}
        }
    }

    /// One CPU cycle.  A period under 2 is ultrasonic (the game's way to silence the channel): the step
    /// stays put rather than whine at the output rate.
    fn clock(&mut self) {
        if self.timer == 0 {
            self.timer = self.period;
            if self.length > 0 && self.linear > 0 && self.period >= 2 {
                self.position = (self.position + 1) & 31;
            }
        } else {
            self.timer -= 1;
        }
    }

    fn quarter_frame(&mut self) {
        if self.reload {
            self.linear = self.reload_value;
        } else if self.linear > 0 {
            self.linear -= 1;
        }
        if !self.control {
            self.reload = false;
        }
    }

    fn half_frame(&mut self) {
        if self.length > 0 && !self.control {
            self.length -= 1;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Noise {
    enabled: bool,
    envelope: Envelope,
    short: bool,
    period: u16,
    timer: u16,
    length: u8,
    shift: u16,
}

impl Default for Noise {
    fn default() -> Self {
        Noise {
            enabled: false,
            envelope: Envelope::default(),
            short: false,
            period: NOISE_PERIODS[0],
            timer: 0,
            length: 0,
            shift: 1,
        }
    }
}

impl Noise {
    fn write(&mut self, register: u8, value: u8) {
        match register {
            0 => self.envelope.write(value),
            2 => {
                self.short = value & 0x80 != 0;
                self.period = NOISE_PERIODS[usize::from(value & 0x0F)];
            }
            3 => {
                if self.enabled {
                    self.length = LENGTHS[usize::from(value >> 3)];
                }
                self.envelope.start = true;
            }
            _ => {}
        }
    }

    /// One CPU cycle.
    fn clock(&mut self) {
        if self.timer == 0 {
            self.timer = self.period - 1;
            let tap = if self.short { 6 } else { 1 };
            let feedback = (self.shift ^ (self.shift >> tap)) & 1;
            self.shift = (self.shift >> 1) | (feedback << 14);
        } else {
            self.timer -= 1;
        }
    }

    fn half_frame(&mut self) {
        if self.length > 0 && !self.envelope.halt {
            self.length -= 1;
        }
    }

    fn output(&self) -> u8 {
        if self.length == 0 || self.shift & 1 != 0 { 0 } else { self.envelope.level() }
    }
}

/// The DMC's output unit.  The machine's DMC fetches the sample bytes (and steals the CPU cycles); this
/// one plays them.
#[derive(Debug, Clone)]
struct Dmc {
    period: u16,
    timer: u16,
    level: u8,
    shifter: u8,
    bits: u8,
    silent: bool,
    /// Fetched bytes not yet played.  The machine holds one; a synth that joined mid-run can be up to an
    /// output cycle out of step with it, hence a queue.
    fetched: VecDeque<u8>,
}

impl Default for Dmc {
    fn default() -> Self {
        Dmc {
            period: DMC_PERIODS[0],
            timer: 0,
            level: 0,
            shifter: 0,
            bits: 8,
            silent: true,
            fetched: VecDeque::new(),
        }
    }
}

impl Dmc {
    /// One CPU cycle.
    fn clock(&mut self) {
        if self.timer > 0 {
            self.timer -= 1;
            return;
        }
        self.timer = self.period - 1;
        if !self.silent {
            if self.shifter & 1 != 0 {
                if self.level <= 125 {
                    self.level += 2;
                }
            } else if self.level >= 2 {
                self.level -= 2;
            }
            self.shifter >>= 1;
        }
        self.bits -= 1;
        if self.bits == 0 {
            self.bits = 8;
            match self.fetched.pop_front() {
                Some(byte) => {
                    self.shifter = byte;
                    self.silent = false;
                }
                None => self.silent = true,
            }
        }
    }
}

/// A first-order high-pass or low-pass filter.
#[derive(Debug, Default, Clone, Copy)]
struct OnePole {
    k: f32,
    input: f32,
    output: f32,
}

impl OnePole {
    fn high(&mut self, x: f32) -> f32 {
        self.output = self.k * (self.output + x - self.input);
        self.input = x;
        self.output
    }

    fn low(&mut self, x: f32) -> f32 {
        self.output += self.k * (x - self.output);
        self.output
    }
}

/// The mixer's output is averaged down to this many times the output rate, then low-passed and decimated.
const OVERSAMPLE: usize = 4;
const TAPS: usize = 64;
/// Output level.  The mixer's full scale is about 1.0, but music uses a fraction of it: eleven minutes of
/// play (the corpus's golden path) peak at 0.38 of full scale after the filters, so 2.0 leaves headroom and
/// the rare louder moment is clipped.
const GAIN: f32 = 2.0;

/// The console's sound from its register writes (module docs): mono samples in `-1.0..=1.0`.
#[derive(Debug, Clone)]
pub struct Synth {
    pulse: [Pulse; 2],
    triangle: Triangle,
    noise: Noise,
    dmc: Dmc,
    /// The frame sequencer: five-step mode, and CPU cycles into the sequence.
    five_step: bool,
    sequence: u32,
    /// An odd CPU cycle (the pulse timers run on every second one).
    odd: bool,
    pulse_levels: [f32; 31],
    tnd_levels: [f32; 203],
    /// Oversampled samples per CPU cycle, and the fraction of one gathered so far.
    step: f64,
    phase: f64,
    sum: f32,
    count: u32,
    /// The decimation filter and the last `TAPS` oversampled samples (a ring, `at` the oldest).
    kernel: [f32; TAPS],
    history: [f32; TAPS],
    at: usize,
    until_output: usize,
    high_90: OnePole,
    high_440: OnePole,
    low_14k: OnePole,
}

impl Synth {
    /// A silent synth producing `rate` samples per second.
    #[must_use]
    pub fn new(rate: f64) -> Synth {
        let mut pulse_levels = [0.0; 31];
        for (n, level) in pulse_levels.iter_mut().enumerate().skip(1) {
            *level = (95.52 / (8128.0 / n as f64 + 100.0)) as f32;
        }
        let mut tnd_levels = [0.0; 203];
        for (n, level) in tnd_levels.iter_mut().enumerate().skip(1) {
            *level = (163.67 / (24329.0 / n as f64 + 100.0)) as f32;
        }
        // A windowed-sinc low-pass at 0.45 of the output rate (Blackman window), unity gain at DC.
        let cutoff = 0.45 / OVERSAMPLE as f64;
        let mut kernel = [0.0f32; TAPS];
        let mut total = 0.0;
        let mid = (TAPS - 1) as f64 / 2.0;
        let mut taps = [0.0f64; TAPS];
        for (i, tap) in taps.iter_mut().enumerate() {
            let x = i as f64 - mid;
            let sinc = if x == 0.0 {
                2.0 * cutoff
            } else {
                (std::f64::consts::TAU * cutoff * x).sin() / (std::f64::consts::PI * x)
            };
            let w = std::f64::consts::TAU * i as f64 / (TAPS - 1) as f64;
            *tap = sinc * (0.42 - 0.5 * w.cos() + 0.08 * (2.0 * w).cos());
            total += *tap;
        }
        for (k, tap) in kernel.iter_mut().zip(taps) {
            *k = (tap / total) as f32;
        }
        let dt = 1.0 / rate;
        let pole = |hz: f64| 1.0 / (std::f64::consts::TAU * hz);
        let mut synth = Synth {
            pulse: [Pulse::default(), Pulse { second: true, ..Pulse::default() }],
            triangle: Triangle::default(),
            noise: Noise::default(),
            dmc: Dmc::default(),
            five_step: false,
            sequence: 0,
            odd: false,
            pulse_levels,
            tnd_levels,
            step: 0.0,
            phase: 0.0,
            sum: 0.0,
            count: 0,
            kernel,
            history: [0.0; TAPS],
            at: 0,
            until_output: OVERSAMPLE,
            high_90: OnePole { k: (pole(90.0) / (pole(90.0) + dt)) as f32, ..OnePole::default() },
            high_440: OnePole { k: (pole(440.0) / (pole(440.0) + dt)) as f32, ..OnePole::default() },
            low_14k: OnePole { k: (dt / (pole(14_000.0) + dt)) as f32, ..OnePole::default() },
        };
        synth.set_rate(rate);
        // The idle channels sit at a constant level (the triangle's first step): start the filters settled
        // on it, so a synth nothing wrote to is silent rather than a fading thump.
        let idle = synth.level();
        synth.history = [idle; TAPS];
        synth.high_90.input = idle;
        synth
    }

    /// Changes the output rate from the next sample on, for a player that keeps a device's buffer level by
    /// nudging the rate a fraction of a percent (the filters keep the rate the synth was made with).
    pub fn set_rate(&mut self, rate: f64) {
        self.step = rate * OVERSAMPLE as f64 / CPU_HZ;
    }

    /// Takes over the sound of a running machine: the registers its program wrote last (those known: none
    /// right after [`ZCore::restore`]), the machine's length counters and frame sequencer.  Envelopes start
    /// over and a triangle note cut by its linear counter stays cut; the DMC plays from its next byte.
    ///
    /// [`ZCore::restore`]: crate::ZCore::restore
    pub fn resume(&mut self, nes: &ZCore) {
        let (tap, apu) = nes.sound_source();
        let before = self.level();
        self.pulse = [Pulse::default(), Pulse { second: true, ..Pulse::default() }];
        self.triangle = Triangle::default();
        self.noise = Noise::default();
        self.dmc = Dmc::default();
        for register in 0..0x14 {
            if let Some(value) = tap.register(register) {
                self.write(register as u8, value);
            }
        }
        // The writes above loaded nothing (no channel was enabled) and left every one-shot flag set.
        let lengths = [&apu.pulse1, &apu.pulse2, &apu.triangle, &apu.noise];
        for (pulse, length) in self.pulse.iter_mut().zip(lengths) {
            (pulse.enabled, pulse.length) = (length.enabled, length.counter);
            pulse.sweep_reload = false;
        }
        (self.triangle.enabled, self.triangle.length) = (apu.triangle.enabled, apu.triangle.counter);
        self.triangle.linear = if self.triangle.control { self.triangle.reload_value } else { 0 };
        self.triangle.reload = self.triangle.control;
        (self.noise.enabled, self.noise.length) = (apu.noise.enabled, apu.noise.counter);
        self.five_step = apu.frame_counter.mode == 1;
        self.sequence = apu.frame_counter.cycle.min(LAST_STEP[usize::from(self.five_step)] - 1);
        // Where the channels' level jumped with the takeover, move the filters' memory with it: no thump.
        let jump = self.level() - before;
        for sample in &mut self.history {
            *sample += jump;
        }
        self.high_90.input += jump;
        self.sum += jump * self.count as f32;
    }

    fn write(&mut self, register: u8, value: u8) {
        match register {
            0x00..=0x03 => self.pulse[0].write(register, value),
            0x04..=0x07 => self.pulse[1].write(register - 4, value),
            0x08..=0x0B => self.triangle.write(register - 8, value),
            0x0C..=0x0F => self.noise.write(register - 0x0C, value),
            0x10 => self.dmc.period = DMC_PERIODS[usize::from(value & 0x0F)],
            0x11 => self.dmc.level = value & 0x7F,
            0x15 => {
                for (bit, pulse) in self.pulse.iter_mut().enumerate() {
                    pulse.enabled = value & (1 << bit) != 0;
                    if !pulse.enabled {
                        pulse.length = 0;
                    }
                }
                self.triangle.enabled = value & 4 != 0;
                if !self.triangle.enabled {
                    self.triangle.length = 0;
                }
                self.noise.enabled = value & 8 != 0;
                if !self.noise.enabled {
                    self.noise.length = 0;
                }
            }
            0x17 => {
                self.five_step = value & 0x80 != 0;
                self.sequence = 0;
                if self.five_step {
                    self.quarter_frame();
                    self.half_frame();
                }
            }
            SAMPLE => self.dmc.fetched.push_back(value),
            _ => {}
        }
    }

    fn quarter_frame(&mut self) {
        self.pulse[0].envelope.clock();
        self.pulse[1].envelope.clock();
        self.noise.envelope.clock();
        self.triangle.quarter_frame();
    }

    fn half_frame(&mut self) {
        self.pulse[0].half_frame();
        self.pulse[1].half_frame();
        self.triangle.half_frame();
        self.noise.half_frame();
    }

    /// The mixer's output for the channels as they stand, `0.0..1.0`.
    fn level(&self) -> f32 {
        let pulse = self.pulse[0].output() + self.pulse[1].output();
        let triangle = TRIANGLE[usize::from(self.triangle.position)];
        let tnd = 3 * usize::from(triangle) + 2 * usize::from(self.noise.output()) + usize::from(self.dmc.level);
        self.pulse_levels[usize::from(pulse)] + self.tnd_levels[tnd]
    }

    /// One oversampled sample: every `OVERSAMPLE`th one makes an output sample.
    fn oversampled(&mut self, sample: f32, out: &mut Vec<f32>) {
        self.history[self.at] = sample;
        self.at = (self.at + 1) % TAPS;
        self.until_output -= 1;
        if self.until_output > 0 {
            return;
        }
        self.until_output = OVERSAMPLE;
        let (newer, older) = self.history.split_at(self.at);
        let mixed: f32 = older.iter().chain(newer).zip(&self.kernel).map(|(x, k)| x * k).sum();
        let filtered = self.low_14k.low(self.high_440.high(self.high_90.high(mixed)));
        out.push((filtered * GAIN).clamp(-1.0, 1.0));
    }

    /// Plays one frame of the machine ([`ZCore::sound`]), appending its samples to `out`: the frame's CPU
    /// cycles at the output rate, so the samples of many frames add up to the time they took.
    ///
    /// [`ZCore::sound`]: crate::ZCore::sound
    pub fn run(&mut self, frame: Frame<'_>, out: &mut Vec<f32>) {
        let mut events = frame.events.iter().peekable();
        for cycle in 0..frame.cycles {
            while let Some(event) = events.next_if(|event| event.cycle <= cycle) {
                self.write(event.register, event.value);
            }
            self.sequence += 1;
            let last = LAST_STEP[usize::from(self.five_step)];
            if self.sequence == last {
                self.quarter_frame();
                self.half_frame();
                self.sequence = 0;
            } else if QUARTERS.contains(&self.sequence) {
                self.quarter_frame();
                if self.sequence == QUARTERS[1] {
                    self.half_frame();
                }
            }
            self.odd = !self.odd;
            if self.odd {
                self.pulse[0].clock();
                self.pulse[1].clock();
            }
            self.triangle.clock();
            self.noise.clock();
            self.dmc.clock();

            self.sum += self.level();
            self.count += 1;
            self.phase += self.step;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                let sample = self.sum / self.count as f32;
                (self.sum, self.count) = (0.0, 0);
                self.oversampled(sample, out);
            }
        }
        // Events timed at or past the frame's end (none in practice) still count.
        for event in events {
            self.write(event.register, event.value);
        }
    }

    /// Plays `cycles` CPU cycles with the register writes `writes` (`(cycle, address, value)`, in order).
    #[cfg(test)]
    fn play(&mut self, writes: &[(u32, u16, u8)], cycles: u32, out: &mut Vec<f32>) {
        let events: Vec<Event> = writes
            .iter()
            .map(|&(cycle, addr, value)| Event { cycle, register: (addr - 0x4000) as u8, value })
            .collect();
        self.run(Frame { events: &events, cycles }, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;
    const SECOND: u32 = CPU_HZ as u32;

    /// How much of `hz` is in the last half second of `samples` (the Goertzel magnitude, per sample).
    fn tone(samples: &[f32], hz: f64) -> f64 {
        let tail = &samples[samples.len() - (RATE / 2.0) as usize..];
        let coefficient = 2.0 * (std::f64::consts::TAU * hz / RATE).cos();
        let (mut s1, mut s2) = (0.0, 0.0);
        for &x in tail {
            (s1, s2) = (f64::from(x) + coefficient * s1 - s2, s1);
        }
        (s1 * s1 + s2 * s2 - coefficient * s1 * s2).sqrt() / tail.len() as f64
    }

    /// `samples` sound the note `hz`: far more of it than of the notes a semitone under and over.
    fn plays(samples: &[f32], hz: f64) -> bool {
        let semitone = 2f64.powf(1.0 / 12.0);
        let (at, under, over) = (tone(samples, hz), tone(samples, hz / semitone), tone(samples, hz * semitone));
        at > 0.01 && at > 8.0 * under.max(over)
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn a_synth_nothing_wrote_to_is_silent_and_keeps_time() {
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        for _ in 0..600 {
            synth.play(&[], 29_781, &mut out);
        }
        assert!(out.iter().all(|&x| x.abs() < 1e-5), "silence, to rounding error");
        let expected = 600.0 * 29_781.0 * RATE / CPU_HZ;
        assert!((out.len() as f64 - expected).abs() <= 1.0, "{} samples for {expected}", out.len());
    }

    #[test]
    fn a_pulse_plays_its_period() {
        // Period 253: 1789773 / (16 * 254) = 440.4 Hz.  Half duty, constant volume 15, length halted.
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        synth.play(&[(0, 0x4015, 0x01), (1, 0x4000, 0xBF), (2, 0x4002, 253), (3, 0x4003, 0x00)], SECOND, &mut out);
        assert!(plays(&out, 440.4) && !plays(&out, 220.2));
        assert!(rms(&out[out.len() / 2..]) > 0.05);
        // The second pulse at twice the period sounds an octave lower.
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        synth.play(&[(0, 0x4015, 0x02), (1, 0x4004, 0xBF), (2, 0x4006, 0xFB), (3, 0x4007, 0x01)], SECOND, &mut out);
        assert!(plays(&out, 220.2) && !plays(&out, 440.4));
    }

    #[test]
    fn a_disabled_channel_and_a_note_whose_length_ran_out_are_silent() {
        let note = [(1, 0x4000, 0xBF), (2, 0x4002, 253), (3, 0x4003, 0x00)];
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        synth.play(&note, SECOND, &mut out);
        assert!(rms(&out[out.len() / 2..]) < 1e-4, "no $4015 enable, no length load");

        // Length index 3 is 2 half frames (1/60 s) when the counter is not halted; constant volume.
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        synth.play(&[(0, 0x4015, 0x01), (1, 0x4000, 0x9F), (2, 0x4002, 253), (3, 0x4003, 0x18)], SECOND, &mut out);
        assert!(rms(&out[..400]) > 0.05, "the note sounds first");
        assert!(rms(&out[out.len() / 2..]) < 1e-4, "then its length counter cuts it");
    }

    #[test]
    fn an_envelope_decays_to_silence() {
        // Envelope period 0: one step per quarter frame, 15 steps (1/16 s), no loop; length 254.
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        synth.play(&[(0, 0x4015, 0x01), (1, 0x4000, 0x80), (2, 0x4002, 253), (3, 0x4003, 0x08)], SECOND, &mut out);
        let early = rms(&out[200..1200]);
        let later = rms(&out[1800..2600]);
        assert!(early > 0.03 && later < early * 0.75 && later > 0.0, "{early} then {later}");
        assert!(rms(&out[out.len() / 2..]) < 1e-4);
    }

    #[test]
    fn the_triangle_plays_an_octave_under_a_pulse_of_its_period() {
        // 1789773 / (32 * 254) = 220.2 Hz; linear counter held by the control flag.
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        synth.play(&[(0, 0x4015, 0x04), (1, 0x4008, 0xFF), (2, 0x400A, 253), (3, 0x400B, 0x00)], SECOND, &mut out);
        assert!(plays(&out, 220.2) && !plays(&out, 110.1));
    }

    #[test]
    fn noise_is_loud_and_has_no_pitch() {
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        synth.play(&[(0, 0x4015, 0x08), (1, 0x400C, 0x3F), (2, 0x400E, 0x04), (3, 0x400F, 0x00)], SECOND, &mut out);
        let tail = &out[out.len() / 2..];
        assert!(rms(tail) > 0.02);
        // A tone's autocorrelation at its period is near 1; noise has no such lag.
        let correlation = |lag: usize| -> f32 {
            let n = tail.len() - lag;
            tail[..n].iter().zip(&tail[lag..]).map(|(a, b)| a * b).sum::<f32>() / (n as f32 * rms(tail).powi(2))
        };
        assert!((20..400).all(|lag| correlation(lag).abs() < 0.5));
    }

    #[test]
    fn dmc_bytes_move_the_output_level() {
        // $FF bytes raise the level by 2 per bit; a direct load sets it.
        let mut synth = Synth::new(RATE);
        let mut out = Vec::new();
        let mut events = vec![Event { cycle: 0, register: 0x10, value: 0x0F }];
        events.extend((0..4).map(|i| Event { cycle: 10 + 432 * i, register: SAMPLE, value: 0xFF }));
        synth.run(Frame { events: &events, cycles: 6000 }, &mut out);
        assert_eq!(synth.dmc.level, 2 * 8 * 4);
        assert!(synth.dmc.silent && synth.dmc.fetched.is_empty());
        synth.play(&[(0, 0x4011, 0x7F)], 10, &mut out);
        assert_eq!(synth.dmc.level, 127);
    }

    #[test]
    fn the_tap_keeps_the_last_writes_until_the_machine_goes_elsewhere_and_logs_only_when_asked() {
        let mut tap = Tap::default();
        tap.begin_frame(1000);
        tap.write(1010, 0x4002, 0x55);
        tap.sample(1020, 0xAA);
        tap.end_frame(30_781);
        assert_eq!((tap.register(2), tap.register(3)), (Some(0x55), None));
        assert!(tap.frame().is_none(), "sound is off");

        tap.set_logging(true);
        tap.begin_frame(u32::MAX - 4);
        tap.write(5, 0x4015, 0x0F);
        tap.sample(7, 0xAA);
        tap.end_frame(29_775);
        let frame = tap.frame().unwrap();
        assert_eq!(frame.cycles(), 29_780, "the cycle counter wraps");
        assert_eq!(frame.events, [
            Event { cycle: 10, register: 0x15, value: 0x0F },
            Event { cycle: 12, register: SAMPLE, value: 0xAA },
        ]);
        tap.begin_frame(29_775);
        assert!(tap.frame().unwrap().events.is_empty(), "a frame's events are its own");

        tap.forget();
        assert_eq!((tap.register(2), tap.register(0x15)), (None, None));
    }

    #[test]
    fn a_higher_rate_makes_proportionally_more_samples() {
        let (mut a, mut b) = (Synth::new(RATE), Synth::new(RATE));
        b.set_rate(RATE * 1.005);
        let (mut slow, mut fast) = (Vec::new(), Vec::new());
        a.play(&[], SECOND, &mut slow);
        b.play(&[], SECOND, &mut fast);
        assert!((fast.len() as f64 - slow.len() as f64 * 1.005).abs() <= 1.0);
    }
}
