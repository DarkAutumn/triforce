//! The numeric kernels. Every result is a pure function of its inputs with a fixed accumulation order: a dot
//! product keeps `LANES` independent partial sums over the input in order and adds them pairwise in a fixed tree,
//! then the remainder in order. Rust never contracts `a * b + c` into a fused multiply-add on its own and these
//! loops have no runtime CPU-feature dispatch, so the bits don't depend on the machine's vector width, the batch
//! or the thread. The lane array is what lets LLVM vectorize the loop without reordering any sum.

const LANES: usize = 8;

/// Σ a[i]·b[i] in the fixed order described above. `a` and `b` have the same length.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; LANES];
    let chunks = a.len() / LANES;
    let (a_main, a_rest) = a.split_at(chunks * LANES);
    let (b_main, b_rest) = b.split_at(chunks * LANES);
    for (x, y) in a_main.chunks_exact(LANES).zip(b_main.chunks_exact(LANES)) {
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
    let chunks = a.len() / LANES;
    let (main, rest) = a.split_at(chunks * LANES);
    for x in main.chunks_exact(LANES) {
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
#[inline]
pub fn linear(w: &[f32], b: &[f32], x: &[f32], y: &mut [f32]) {
    let n_in = x.len();
    debug_assert_eq!(w.len(), y.len() * n_in);
    for (o, out) in y.iter_mut().enumerate() {
        let bias = if b.is_empty() { 0.0 } else { b[o] };
        *out = dot(&w[o * n_in..(o + 1) * n_in], x) + bias;
    }
}

/// In-place LayerNorm over `x` (biased variance, eps 1e-5, as torch), then `x·gamma + beta`.
#[inline]
pub fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len() as f32;
    let mean = sum(x) / n;
    let mut var_acc = [0f32; LANES];
    let chunks = x.len() / LANES;
    let (main, rest) = x.split_at(chunks * LANES);
    for c in main.chunks_exact(LANES) {
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
}
