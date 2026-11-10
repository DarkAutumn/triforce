//! Keyed Gumbel noise (Python: `triforce._native.keyed_noise`, which tests feed to the torch twin
//! `ZeldaPolicy.evaluate(noise=)`): a pure function of (job, decision step, column), counter-based, so a decision's
//! noise doesn't depend on which worker draws it or when. A play's actions can be sampled again exactly from its job
//! key. Every uniform behind it lies in `[2^-54, 1 - 2^-53]`, so every Gumbel value is finite.

const NOISE_TAG: u64 = 0x7F4A_7C15_9E37_79B9;

/// SplitMix64's finalizer.
#[inline]
pub fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Standard Gumbel noise for decision `step` of job `job`, `out.len()` columns.
pub fn keyed_noise(job: i64, step: i64, out: &mut [f32]) {
    let key = mix(mix(job as u64 ^ NOISE_TAG).wrapping_add((step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    gumbel(key, out);
}

/// A key from a sequence of parts (search: job, decision, tag, visit, future, ...), order-sensitive.
pub fn key_of(parts: &[u64]) -> u64 {
    parts.iter().fold(NOISE_TAG, |h, &p| mix(h ^ mix(p.wrapping_add(0x9E37_79B9_7F4A_7C15))))
}

/// The largest uniform: `x + 0.5` with the top 53-bit `x` rounds to 2^53 in f64, i.e. a uniform of exactly 1.0
/// (a Gumbel of +inf); clamped to the value below.
const UNIFORM_MAX: f64 = 1.0 - f64::EPSILON / 2.0;

/// A uniform in (0, 1) from `key` alone (`[2^-54, 1 - 2^-53]`: never 0 or 1, so every Gumbel is finite).
pub fn uniform(key: u64) -> f64 {
    (((mix(key) >> 11) as f64 + 0.5) * 2f64.powi(-53)).min(UNIFORM_MAX)
}

/// Standard Gumbel noise from `key` alone, `out.len()` columns ([`keyed_noise`] is this with a (job, step) key).
pub fn gumbel(key: u64, out: &mut [f32]) {
    for (c, o) in out.iter_mut().enumerate() {
        let uniform = uniform(key ^ (c as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        *o = (-(-uniform.ln()).ln()) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_keyed_and_finite() {
        let (mut a, mut b, mut c) = ([0f32; 19], [0f32; 19], [0f32; 19]);
        keyed_noise(7, 3, &mut a);
        keyed_noise(7, 3, &mut b);
        keyed_noise(7, 4, &mut c);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    /// The inverse of [`mix`] (each xorshift and odd multiply is invertible).
    fn unmix(x: u64) -> u64 {
        fn unshift(y: u64, s: u32) -> u64 {
            let mut x = y;
            for _ in 0..(64 / s) {
                x = y ^ (x >> s);
            }
            x
        }
        fn inverse(a: u64) -> u64 {
            let mut x = a; // Newton: 5 steps reach 64 bits from 3
            for _ in 0..5 {
                x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
            }
            x
        }
        let x = unshift(x, 31).wrapping_mul(inverse(0x94D0_49BB_1331_11EB));
        let x = unshift(x, 27).wrapping_mul(inverse(0xBF58_476D_1CE4_E5B9));
        unshift(x, 30)
    }

    #[test]
    fn uniform_extremes_stay_inside_and_gumbel_finite() {
        for top in [u64::MAX, 0] {
            assert_eq!(mix(unmix(top)), top);
        }
        let high = uniform(unmix(u64::MAX));
        let low = uniform(unmix(0));
        assert!(high < 1.0 && low > 0.0, "{low} {high}");
        for top in [u64::MAX, 0] {
            let mut out = [0f32; 1];
            gumbel(unmix(top), &mut out); // column 0: key ^ 0
            assert!(out[0].is_finite(), "{top:#x} -> {}", out[0]);
        }
    }
}
