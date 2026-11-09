//! Keyed Gumbel noise (`triforce/inference.py` `keyed_noise`): a pure function of (job, decision step, column),
//! counter-based, so a decision's noise doesn't depend on which worker draws it or when. A play's actions can be
//! sampled again exactly from its job key.

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
    for (c, o) in out.iter_mut().enumerate() {
        let bits = mix(key ^ (c as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let uniform = ((bits >> 11) as f64 + 0.5) * 2f64.powi(-53);
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
}
