//! The game's random number register, `Random` at `$18-$24` (zelda-asm `mechanics.md` "# Random numbers").
//!
//! One 13-byte shift register steps once per NMI frame (asm:Nmi_Isr@ScrambleRandom): the new bit is bit 1 of
//! `$18` XOR bit 1 of `$19`, then the 13 bytes rotate right as one 104-bit value, the new bit entering bit 7 of
//! `$18`.  Reset seeds it with `$40` in `$18` and 0 in the other 12 bytes (asm:Reset_ClearRam), and nothing
//! else ever writes it.  The step drops bit 0 of `$24` without reading it, so it is not invertible: from the
//! seed the register runs a lead-in of [`LEAD_IN`] states that never come back, then cycles with period
//! [`PERIOD`].  Those 32,767 states ([`PHASES`]) are the only values a game in play can hold.
//!
//! A start from a saved state writes one of them ([`write`]): the register after `phase` steps from the seed,
//! the phase drawn uniformly from [`PHASES`] by the caller's seeded generator ([`draw_phase`]).  Stepping is a
//! few integer operations on a `u128`; [`natural`] steps at most 32,854 times (tens of microseconds, against a
//! start-state load's milliseconds of emulated entry frames), so no table is kept.

use crate::nes::Nes;
use crate::vars::RNG_VARS;
use rand::{Rng, RngExt};
use std::ops::RangeInclusive;

/// The register's address (`$18`) and length.
pub const ADDR: u16 = 0x18;
pub const LEN: usize = 13;
/// The register after reset (asm:Reset_ClearRam).
pub const SEED: [u8; LEN] = [0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
/// Steps from the seed before the cycle: the states after 0..LEAD_IN steps occur once, after power on.
pub const LEAD_IN: u32 = 88;
/// The cycle's length: the state after `n + PERIOD` steps is the state after `n` for every `n >= LEAD_IN`.
pub const PERIOD: u32 = 32767;
/// The phases of the cycle's states, each once: what a start draws from.
pub const PHASES: RangeInclusive<u32> = LEAD_IN..=LEAD_IN + PERIOD - 1;

const BITS: u32 = 8 * LEN as u32;

/// The register as one 104-bit number, `$18` the most significant byte.
fn pack(bytes: &[u8; LEN]) -> u128 {
    bytes.iter().fold(0, |v, &b| v << 8 | b as u128)
}

fn unpack(mut v: u128) -> [u8; LEN] {
    let mut out = [0; LEN];
    for b in out.iter_mut().rev() {
        *b = v as u8;
        v >>= 8;
    }
    out
}

/// One frame's step (asm:Nmi_Isr@ScrambleRandom): bit 1 of `$18` is bit 97, bit 1 of `$19` bit 89.
#[inline]
fn step_packed(v: u128) -> u128 {
    let new = ((v >> 97) ^ (v >> 89)) & 1;
    v >> 1 | new << (BITS - 1)
}

/// The register one frame later.
pub fn step(bytes: &[u8; LEN]) -> [u8; LEN] {
    unpack(step_packed(pack(bytes)))
}

/// The register `steps` frames after reset ([`SEED`] stepped `steps` times).
pub fn natural(steps: u32) -> [u8; LEN] {
    let mut v = pack(&SEED);
    for _ in 0..steps {
        v = step_packed(v);
    }
    unpack(v)
}

/// Whether the game can hold `bytes` in play: stepping them [`PERIOD`] times returns them.
pub fn on_cycle(bytes: &[u8; LEN]) -> bool {
    let start = pack(bytes);
    let mut v = start;
    for _ in 0..PERIOD {
        v = step_packed(v);
    }
    v == start
}

/// The number of steps from reset that gives `bytes`, the smallest (`0..LEAD_IN` for a lead-in state,
/// [`PHASES`] for a state of the cycle); None for a value the game never holds.
pub fn phase_of(bytes: &[u8; LEN]) -> Option<u32> {
    let target = pack(bytes);
    let mut v = pack(&SEED);
    for n in 0..=*PHASES.end() {
        if v == target {
            return Some(n);
        }
        v = step_packed(v);
    }
    None
}

/// A phase of the cycle, uniform over [`PHASES`].
pub fn draw_phase<R: Rng + ?Sized>(rng: &mut R) -> u32 {
    rng.random_range(PHASES)
}

/// The register as the machine holds it now.
pub fn read<N: Nes>(nes: &N) -> [u8; LEN] {
    RNG_VARS.map(|var| nes.peek(var.addr()))
}

/// Writes the register `phase` steps after reset (one poke per byte, so a recording keeps them as edits).
pub fn write<N: Nes>(nes: &mut N, phase: u32) {
    for (var, value) in RNG_VARS.into_iter().zip(natural(phase)) {
        nes.poke(var.addr(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn the_cycle_starts_after_the_lead_in() {
        // Round 22's reading of the register: frame 88's state, the first of the cycle.
        assert_eq!(natural(LEAD_IN), [0x9C, 0x8B, 0xB2, 0xA5, 0xC0, 0x8B, 0x0A, 0x1C, 0x08, 0x30, 0x20, 0x40, 0x00]);
        assert_eq!(natural(LEAD_IN + PERIOD), natural(LEAD_IN));
        assert_ne!(natural(LEAD_IN - 1 + PERIOD), natural(LEAD_IN - 1), "87 is still the lead-in");
        assert!(on_cycle(&natural(LEAD_IN)) && on_cycle(&natural(*PHASES.end())));
        assert!(!on_cycle(&natural(LEAD_IN - 1)) && !on_cycle(&SEED));
        // No shorter period: the cycle's states are all different.
        let mut seen = std::collections::HashSet::new();
        let mut v = natural(LEAD_IN);
        for _ in PHASES {
            assert!(seen.insert(v));
            v = step(&v);
        }
        assert_eq!(v, natural(LEAD_IN));
    }

    #[test]
    fn phases_read_back() {
        for n in [0, 1, 87, 88, 89, 1000, 32854] {
            assert_eq!(phase_of(&natural(n)), Some(n));
        }
        assert_eq!(phase_of(&natural(32855)), Some(88));
        assert_eq!(phase_of(&[0xFF; LEN]), None);
        assert!(!on_cycle(&[0xFF; LEN]));
    }

    #[test]
    fn draws_cover_the_cycle_and_follow_the_seed() {
        let draw = |seed| draw_phase(&mut rand_chacha::ChaCha8Rng::seed_from_u64(seed));
        assert_eq!(draw(5), draw(5));
        assert_ne!(draw(5), draw(6));
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let phases: Vec<u32> = (0..2000).map(|_| draw_phase(&mut rng)).collect();
        assert!(phases.iter().all(|p| PHASES.contains(p)));
        assert!(phases.iter().any(|&p| p < 1000) && phases.iter().any(|&p| p > 31854));
    }
}
