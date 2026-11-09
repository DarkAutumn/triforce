//! The numeric kernels. Every result is a pure function of its inputs with a fixed accumulation order: a dot
//! product keeps `LANES` independent partial sums over the input in order and adds them pairwise in a fixed tree,
//! then the remainder in order. Rust never contracts `a * b + c` into a fused multiply-add on its own and the
//! explicit AVX paths do exactly the portable code's products and sums per lane (no FMA), so the bits don't depend
//! on the machine's vector width, the path taken, the batch or the thread. The lane array is what lets LLVM
//! vectorize the portable loops without reordering any sum.

const LANES: usize = 8;

/// Σ a[i]·b[i] in the fixed order described above. `a` and `b` have the same length.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; LANES];
    let (a_main, a_rest) = a.as_chunks::<LANES>();
    let (b_main, b_rest) = b.as_chunks::<LANES>();
    for (x, y) in a_main.iter().zip(b_main) {
        for l in 0..LANES {
            acc[l] += x[l] * y[l];
        }
    }
    let mut sum = reduce(&acc);
    for (x, y) in a_rest.iter().zip(b_rest) {
        sum += x * y;
    }
    sum
}

/// Σ a[i] in the same fixed order as [`dot`].
#[inline]
pub fn sum(a: &[f32]) -> f32 {
    let mut acc = [0f32; LANES];
    let (main, rest) = a.as_chunks::<LANES>();
    for x in main {
        for l in 0..LANES {
            acc[l] += x[l];
        }
    }
    let mut total = reduce(&acc);
    for x in rest {
        total += x;
    }
    total
}

#[inline]
fn reduce(acc: &[f32; LANES]) -> f32 {
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// y = W·x + b: `w` is `[out, in]` row-major (torch's `Linear.weight`), `b` has `out` entries (or is empty).
/// Each output is `dot(w_row, x) + bias`, bit for bit, on either path.
#[inline]
pub fn linear(w: &[f32], b: &[f32], x: &[f32], y: &mut [f32]) {
    let n_in = x.len();
    assert_eq!(w.len(), y.len() * n_in);
    assert!(b.is_empty() || b.len() == y.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") {
        // SAFETY: AVX is available; the lengths are checked above.
        return unsafe { linear_avx(w, b, x, y) };
    }
    linear_portable(w, b, x, y);
}

/// [`linear`] without AVX.
fn linear_portable(w: &[f32], b: &[f32], x: &[f32], y: &mut [f32]) {
    let n_in = x.len();
    for (o, out) in y.iter_mut().enumerate() {
        let bias = if b.is_empty() { 0.0 } else { b[o] };
        *out = dot(&w[o * n_in..(o + 1) * n_in], x) + bias;
    }
}

/// [`linear`] with AVX, four output rows at a time (one load of `x` per chunk): per row exactly [`dot`]'s lanes,
/// tree and remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn linear_avx(w: &[f32], b: &[f32], x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::{__m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps};
    const ROWS: usize = 4;
    let n_in = x.len();
    let chunks = n_in / LANES;
    let finish = |acc: __m256, o: usize| -> f32 {
        let mut lanes = [0f32; LANES];
        // SAFETY: `lanes` holds 8 floats.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut sum = reduce(&lanes);
        for (wv, xv) in w[o * n_in + chunks * LANES..(o + 1) * n_in].iter().zip(&x[chunks * LANES..]) {
            sum += wv * xv;
        }
        sum + if b.is_empty() { 0.0 } else { b[o] }
    };
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    let mut o = 0;
    // SAFETY (loads): row `o + r` < y.len() and chunk `k` < n_in / 8, so every load is inside `w` and `x`.
    while o + ROWS <= y.len() {
        let mut acc = [_mm256_setzero_ps(); ROWS];
        for k in 0..chunks {
            let xv = unsafe { _mm256_loadu_ps(xp.add(k * LANES)) };
            for (r, a) in acc.iter_mut().enumerate() {
                let wv = unsafe { _mm256_loadu_ps(wp.add((o + r) * n_in + k * LANES)) };
                *a = _mm256_add_ps(*a, _mm256_mul_ps(wv, xv));
            }
        }
        for (r, a) in acc.into_iter().enumerate() {
            y[o + r] = finish(a, o + r);
        }
        o += ROWS;
    }
    while o < y.len() {
        let mut acc = _mm256_setzero_ps();
        for k in 0..chunks {
            let (wv, xv) = unsafe { (_mm256_loadu_ps(wp.add(o * n_in + k * LANES)), _mm256_loadu_ps(xp.add(k * LANES))) };
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
        }
        y[o] = finish(acc, o);
        o += 1;
    }
}

/// In-place LayerNorm over `x` (biased variance, eps 1e-5, as torch), then `x·gamma + beta`.
#[inline]
pub fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len() as f32;
    let mean = sum(x) / n;
    let mut var_acc = [0f32; LANES];
    let (main, rest) = x.as_chunks::<LANES>();
    for c in main {
        for l in 0..LANES {
            let d = c[l] - mean;
            var_acc[l] += d * d;
        }
    }
    let mut var = reduce(&var_acc);
    for v in rest {
        let d = v - mean;
        var += d * d;
    }
    let inv = 1.0 / (var / n + 1e-5).sqrt();
    for ((v, g), b) in x.iter_mut().zip(gamma).zip(beta) {
        *v = (*v - mean) * inv * g + b;
    }
}

/// acc[i] += a·x[i] for every i (one product and one sum per element: no order to fix).
#[inline]
pub fn axpy(acc: &mut [f32], a: f32, x: &[f32]) {
    debug_assert_eq!(acc.len(), x.len());
    let (acc_main, acc_rest) = acc.as_chunks_mut::<LANES>();
    let (x_main, x_rest) = x.as_chunks::<LANES>();
    for (c, v) in acc_main.iter_mut().zip(x_main) {
        for l in 0..LANES {
            c[l] += a * v[l];
        }
    }
    for (c, v) in acc_rest.iter_mut().zip(x_rest) {
        *c += a * v;
    }
}

#[inline]
pub fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

#[inline]
pub fn tanh(x: &mut [f32]) {
    for v in x {
        *v = v.tanh();
    }
}

/// log-softmax of `logits` with masked entries forced to `masked` first (torch `masked_fill` then
/// `log_softmax`): `out[i] = l[i] - max - ln Σ exp(l[j] - max)`.
pub fn masked_log_softmax(logits: &[f32], mask: &[bool], masked: f32, out: &mut [f32]) {
    for ((o, &l), &m) in out.iter_mut().zip(logits).zip(mask) {
        *o = if m { l } else { masked };
    }
    let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0f32;
    for &v in out.iter() {
        total += (v - max).exp();
    }
    let lse = max + total.ln();
    for v in out.iter_mut() {
        *v -= lse;
    }
}

/// Index of the first maximum (torch `argmax`).
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_a_naive_sum_closely_and_is_order_fixed() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.11).cos()).collect();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
        assert!((dot(&a, &b) as f64 - naive).abs() < 1e-5);
        // The same inputs inside a longer buffer give the same bits (no dependence on alignment).
        let mut padded = vec![0f32; 3];
        padded.extend_from_slice(&a);
        assert_eq!(dot(&padded[3..], &b).to_bits(), dot(&a, &b).to_bits());
    }

    #[test]
    fn masked_log_softmax_ignores_masked_entries() {
        let mut out = [0f32; 4];
        masked_log_softmax(&[1.0, 5.0, 2.0, 0.5], &[true, false, true, true], -1e9, &mut out);
        let p: f32 = out.iter().map(|v| v.exp()).sum();
        assert!((p - 1.0).abs() < 1e-6);
        assert_eq!(out[1].exp(), 0.0);
        assert_eq!(argmax(&out), 2);
    }

    #[test]
    fn layer_norm_normalizes() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        layer_norm(&mut x, &[1.0; 10], &[0.0; 10]);
        assert!(sum(&x).abs() < 1e-5);
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 10.0;
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn linear_paths_give_the_same_bits() {
        // Row counts and widths that exercise the 4-row blocks, single rows and the lane remainder.
        for (rows, n_in) in [(1, 3), (4, 8), (7, 37), (13, 64), (256, 425)] {
            let w: Vec<f32> = (0..rows * n_in).map(|i| (i as f32 * 0.731).sin()).collect();
            let b: Vec<f32> = (0..rows).map(|i| (i as f32 * 1.7).cos()).collect();
            let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.29).cos() * 3.0).collect();
            for bias in [&b[..], &[]] {
                let (mut fast, mut portable) = (vec![0f32; rows], vec![0f32; rows]);
                linear(&w, bias, &x, &mut fast);
                linear_portable(&w, bias, &x, &mut portable);
                assert_eq!(fast.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                           portable.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{rows}x{n_in}");
            }
        }
    }
}
