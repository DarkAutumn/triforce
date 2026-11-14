//! Act modes: how an action is chosen from the policy outside training (docs/rust-actor.md "Act modes").
//!
//! `sample` draws from the policy (each head by Gumbel-argmax, the kind first: exactly the joint distribution);
//! `argmax` takes each head's most likely entry (the kind, then the direction and duration given it). The other
//! modes reshape the **joint** distribution over (kind, direction, duration) before drawing from it: the model
//! defines one (p(kind) · p(direction | kind) · p(duration | kind), [`crate::Policy::joint_log_probs`]), so "this
//! action as a whole is unlikely" is a statement about the joint probability, which a per-head rule would miss (two
//! 20% heads make a 4% action). The transforms apply in this order, each to the previous one's result,
//! renormalised:
//!
//! | Spelling | Rule |
//! |---|---|
//! | `temperature:T` | q ∝ p^(1/T) (T < 1 sharpens) |
//! | `floor:P` | drop actions with q < P, never the most probable one |
//! | `minp:R` | drop actions with q < R · max q |
//! | `topp:P` | keep the fewest most probable actions whose q sum to at least P (ties: lower index first) |
//! | `escape:N` | argmax while the env reports fewer than N decisions in a row without progress (its linger count); from N on, sample with the other rules (plain `sample` without any) until progress resets the count |
//! | `relax:N` | the other rules' draw while the env reports fewer than N decisions in a row without progress; from N on, plain `sample` (the policy itself) until progress resets the count; needs another rule |
//!
//! So with a stall rule a mode plays one way until stalled and another while stalled:
//!
//! | Spelling | not stalled | stalled (N on) |
//! |---|---|---|
//! | `escape:N` | argmax | `sample` |
//! | `temperature:T+escape:N` | argmax | a `temperature:T` draw |
//! | `temperature:T+relax:N` | a `temperature:T` draw | `sample` |
//!
//! Rules combine with `+` (`temperature:0.7+floor:0.02`, `escape:16+floor:0.05`); each at most once, and at most one
//! of `escape` and `relax`; `sample` and `argmax` stand alone. [`ActMode::to_string`] is the canonical spelling (rules
//! in the table's order). A rule's draw is Gumbel-argmax over the joint actions with noise keyed by (job, decision)
//! ([`joint_noise_key`]), so a mode plays the same for the same seed wherever it runs.

use std::fmt;
use std::str::FromStr;

/// A transform of the joint distribution before a draw; all `None` is the policy itself.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rule {
    pub temperature: Option<f32>,
    pub floor: Option<f32>,
    pub minp: Option<f32>,
    pub topp: Option<f32>,
}

impl Rule {
    /// No transform: draw from the policy as it is.
    pub fn is_plain(&self) -> bool {
        *self == Rule::default()
    }

    /// The acting probabilities `q` (0 for removed or illegal actions) of joint log-probs `log_probs` (`-inf`:
    /// illegal; at least one finite). `order` is working memory of the same length (no allocation).
    pub fn probabilities(&self, log_probs: &[f32], q: &mut [f64], order: &mut [usize]) {
        let n = log_probs.len();
        debug_assert!(q.len() == n && order.len() == n);
        let inv_t = self.temperature.map_or(1.0, |t| 1.0 / t as f64);
        let max = log_probs
            .iter()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64 * inv_t));
        for (q, &lp) in q.iter_mut().zip(log_probs) {
            *q = if lp.is_finite() { (lp as f64 * inv_t - max).exp() } else { 0.0 };
        }
        normalise(q);
        let best = first_max(q);
        if let Some(p) = self.floor {
            let p = p as f64;
            for (i, v) in q.iter_mut().enumerate() {
                if *v < p && i != best {
                    *v = 0.0;
                }
            }
            normalise(q);
        }
        if let Some(r) = self.minp {
            let cut = r as f64 * q[best];
            for (i, v) in q.iter_mut().enumerate() {
                if *v < cut && i != best {
                    *v = 0.0;
                }
            }
            normalise(q);
        }
        if let Some(p) = self.topp {
            for (i, o) in order.iter_mut().enumerate() {
                *o = i;
            }
            order.sort_by(|&a, &b| q[b].total_cmp(&q[a]).then(a.cmp(&b)));
            let mut total = 0.0;
            let mut keep = 0;
            for &i in order.iter() {
                if q[i] <= 0.0 {
                    break;
                }
                total += q[i];
                keep += 1;
                if total >= p as f64 {
                    break;
                }
            }
            for &i in &order[keep.max(1)..] {
                q[i] = 0.0;
            }
            normalise(q);
        }
    }

    /// The draw from `q` ([`Rule::probabilities`]) with Gumbel noise `noise` (one value per joint action): the
    /// index of the largest `ln q + g` among the kept actions.
    pub fn draw(q: &[f64], noise: &[f32]) -> usize {
        let mut best = usize::MAX;
        let mut score = f64::NEG_INFINITY;
        for (i, (&v, &g)) in q.iter().zip(noise).enumerate() {
            if v > 0.0 {
                let s = v.ln() + g as f64;
                if best == usize::MAX || s > score {
                    best = i;
                    score = s;
                }
            }
        }
        best
    }

    fn fmt_parts(&self, parts: &mut Vec<String>) {
        let mut push = |name: &str, v: Option<f32>| {
            if let Some(v) = v {
                parts.push(format!("{name}:{v}"));
            }
        };
        push("temperature", self.temperature);
        push("floor", self.floor);
        push("minp", self.minp);
        push("topp", self.topp);
    }
}

/// The entropy (nats) of probabilities `q`.
pub fn entropy(q: &[f64]) -> f64 {
    q.iter().filter(|&&v| v > 0.0).map(|&v| -v * v.ln()).sum()
}

fn normalise(q: &mut [f64]) {
    let total: f64 = q.iter().sum();
    if total > 0.0 {
        for v in q.iter_mut() {
            *v /= total;
        }
    }
}

fn first_max(q: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in q.iter().enumerate() {
        if v > q[best] {
            best = i;
        }
    }
    best
}

/// How a policy acts outside training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActMode {
    /// Each head's most likely entry.
    Argmax,
    /// A draw from the policy reshaped by the rule (plain: the policy itself, per head as in training).
    Sample(Rule),
    /// Argmax until `after` decisions in a row without progress, then a draw by the rule until progress.
    Escape { after: u32, rule: Rule },
    /// A draw by the rule (not plain) until `after` decisions in a row without progress, then a draw from the policy
    /// itself (`sample`) until progress.
    Relax { after: u32, rule: Rule },
}

/// How one decision is taken: by argmax, or drawn by a rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Choose {
    Argmax,
    Draw(Rule),
}

impl ActMode {
    pub const SAMPLE: ActMode = ActMode::Sample(Rule {
        temperature: None,
        floor: None,
        minp: None,
        topp: None,
    });

    /// How the decision after `stall` decisions in a row without progress is taken.
    pub fn choose(&self, stall: u32) -> Choose {
        match *self {
            ActMode::Argmax => Choose::Argmax,
            ActMode::Sample(rule) => Choose::Draw(rule),
            ActMode::Escape { after, rule } => {
                if stall >= after {
                    Choose::Draw(rule)
                } else {
                    Choose::Argmax
                }
            }
            ActMode::Relax { after, rule } => Choose::Draw(if stall >= after { Rule::default() } else { rule }),
        }
    }
}

impl Default for ActMode {
    fn default() -> Self {
        ActMode::SAMPLE
    }
}

impl fmt::Display for ActMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        match self {
            ActMode::Argmax => return f.write_str("argmax"),
            ActMode::Sample(rule) if rule.is_plain() => return f.write_str("sample"),
            ActMode::Sample(rule) => rule.fmt_parts(&mut parts),
            ActMode::Escape { after, rule } => {
                rule.fmt_parts(&mut parts);
                parts.push(format!("escape:{after}"));
            }
            ActMode::Relax { after, rule } => {
                rule.fmt_parts(&mut parts);
                parts.push(format!("relax:{after}"));
            }
        }
        f.write_str(&parts.join("+"))
    }
}

impl FromStr for ActMode {
    type Err = String;

    fn from_str(text: &str) -> Result<ActMode, String> {
        let text = text.trim();
        match text {
            "sample" => return Ok(ActMode::SAMPLE),
            "argmax" => return Ok(ActMode::Argmax),
            "" => return Err("empty act mode".to_owned()),
            _ => {}
        }
        let mut rule = Rule::default();
        // The stall rule: ("escape" | "relax", N).
        let mut stall: Option<(&str, u32)> = None;
        for part in text.split('+') {
            let (name, value) = part
                .split_once(':')
                .ok_or_else(|| format!("act mode {text:?}: {part:?} is not sample, argmax or NAME:VALUE"))?;
            let number = |lo_open: f32, hi: f32, inclusive_hi: bool| -> Result<f32, String> {
                let v: f32 = value
                    .parse()
                    .map_err(|_| format!("act mode {text:?}: {name} needs a number, got {value:?}"))?;
                let ok = v.is_finite() && v > lo_open && (v < hi || (inclusive_hi && v == hi));
                if ok {
                    Ok(v)
                } else {
                    Err(format!(
                        "act mode {text:?}: {name} must be in ({lo_open}, {hi}{}",
                        if inclusive_hi { "]" } else { ")" }
                    ))
                }
            };
            let slot = match name {
                "temperature" => &mut rule.temperature,
                "floor" => &mut rule.floor,
                "minp" => &mut rule.minp,
                "topp" => &mut rule.topp,
                "escape" | "relax" => {
                    if let Some((given, _)) = stall {
                        return Err(if given == name {
                            format!("act mode {text:?}: {name} given twice")
                        } else {
                            format!("act mode {text:?}: escape and relax exclude each other")
                        });
                    }
                    let n: u32 = value
                        .parse()
                        .map_err(|_| format!("act mode {text:?}: {name} needs a whole number, got {value:?}"))?;
                    if n == 0 {
                        return Err(format!("act mode {text:?}: {name} must be at least 1"));
                    }
                    stall = Some((name, n));
                    continue;
                }
                _ => {
                    return Err(format!(
                        "act mode {text:?}: unknown rule {name:?} (sample, argmax, temperature:T, floor:P, minp:R, \
                         topp:P, escape:N, relax:N)"
                    ));
                }
            };
            if slot.is_some() {
                return Err(format!("act mode {text:?}: {name} given twice"));
            }
            *slot = Some(match name {
                "temperature" => number(0.0, f32::INFINITY, false)?,
                "floor" => number(0.0, 1.0, false)?,
                _ => number(0.0, 1.0, true)?,
            });
        }
        Ok(match stall {
            Some(("escape", after)) => ActMode::Escape { after, rule },
            Some((_, after)) if rule.is_plain() => {
                return Err(format!(
                    "act mode {text:?}: relax:{after} needs another rule (alone it is sample at every decision)"
                ));
            }
            Some((_, after)) => ActMode::Relax { after, rule },
            None => ActMode::Sample(rule),
        })
    }
}

/// The key of decision `step` of job `job`'s joint-action noise (a rule's draw; `noise::gumbel` over the joint
/// actions), apart from the per-head noise `sample` uses.
pub fn joint_noise_key(job: i64, step: i64) -> u64 {
    crate::noise::key_of(&[JOINT_TAG, job as u64, step as u64])
}

const JOINT_TAG: u64 = 0x4A4F_494E_545F_5251;

#[cfg(test)]
mod tests {
    use super::*;

    fn probs(rule: Rule, lp: &[f32]) -> Vec<f64> {
        let mut q = vec![0.0; lp.len()];
        let mut order = vec![0; lp.len()];
        rule.probabilities(lp, &mut q, &mut order);
        q
    }

    fn ln(p: &[f64]) -> Vec<f32> {
        p.iter().map(|&v| if v > 0.0 { v.ln() as f32 } else { f32::NEG_INFINITY }).collect()
    }

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }

    const P: [f64; 6] = [0.5, 0.3, 0.1, 0.06, 0.04, 0.0];

    #[test]
    fn plain_is_the_policy() {
        close(&probs(Rule::default(), &ln(&P)), &P);
    }

    #[test]
    fn temperature_raises_to_one_over_t() {
        let rule = Rule { temperature: Some(0.5), ..Rule::default() };
        let sq: Vec<f64> = P.iter().map(|p| p * p).collect();
        let total: f64 = sq.iter().sum();
        close(&probs(rule, &ln(&P)), &sq.iter().map(|v| v / total).collect::<Vec<_>>());
    }

    #[test]
    fn floor_drops_below_p_and_keeps_the_best() {
        let rule = Rule { floor: Some(0.08), ..Rule::default() };
        close(&probs(rule, &ln(&P)), &[0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0.0, 0.0, 0.0]);
        // A floor above every action keeps the most probable one alone.
        let rule = Rule { floor: Some(0.9), ..Rule::default() };
        close(&probs(rule, &ln(&P)), &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Even when the most probable is under the floor (a flat distribution) and ties go to the lower index.
        let flat = [0.25; 4];
        close(&probs(Rule { floor: Some(0.5), ..Rule::default() }, &ln(&flat)), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn minp_is_relative_to_the_best() {
        let rule = Rule { minp: Some(0.15), ..Rule::default() };
        close(&probs(rule, &ln(&P)), &[0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0.0, 0.0, 0.0]);
        let rule = Rule { minp: Some(1.0), ..Rule::default() };
        close(&probs(rule, &ln(&P)), &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn topp_keeps_the_fewest_reaching_p() {
        let rule = Rule { topp: Some(0.75), ..Rule::default() };
        close(&probs(rule, &ln(&P)), &[0.5 / 0.8, 0.3 / 0.8, 0.0, 0.0, 0.0, 0.0]);
        let rule = Rule { topp: Some(0.85), ..Rule::default() };
        close(&probs(rule, &ln(&P)), &[0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0.0, 0.0, 0.0]);
        let rule = Rule { topp: Some(1.0), ..Rule::default() };
        close(&probs(rule, &ln(&P)), &P);
        let rule = Rule { topp: Some(0.01), ..Rule::default() };
        close(&probs(rule, &ln(&P)), &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rules_compose_in_order() {
        // temperature first, then the floor on the sharpened distribution
        let rule = Rule { temperature: Some(0.5), floor: Some(0.05), ..Rule::default() };
        let sq: Vec<f64> = P.iter().map(|p| p * p).collect();
        let total: f64 = sq.iter().sum();
        let t: Vec<f64> = sq.iter().map(|v| v / total).collect();
        let kept: f64 = t.iter().filter(|&&v| v >= 0.05).sum();
        let want: Vec<f64> = t.iter().map(|&v| if v >= 0.05 { v / kept } else { 0.0 }).collect();
        close(&probs(rule, &ln(&P)), &want);
    }

    #[test]
    fn draws_respect_the_mask_and_the_rule() {
        let rule = Rule { floor: Some(0.08), ..Rule::default() };
        let q = probs(rule, &ln(&P));
        let mut counts = [0u32; 6];
        let mut noise = [0f32; 6];
        for step in 0..20_000 {
            crate::noise::gumbel(joint_noise_key(3, step), &mut noise);
            counts[Rule::draw(&q, &noise)] += 1;
        }
        assert_eq!(&counts[3..], &[0, 0, 0]);
        for (i, &c) in counts[..3].iter().enumerate() {
            let share = c as f64 / 20_000.0;
            assert!((share - q[i]).abs() < 0.015, "{i}: {share} vs {}", q[i]);
        }
    }

    #[test]
    fn spellings_parse_to_canonical_form() {
        for (text, canonical) in [
            ("sample", "sample"),
            ("argmax", "argmax"),
            ("temperature:0.75", "temperature:0.75"),
            ("floor:0.02+temperature:0.7", "temperature:0.7+floor:0.02"),
            ("minp:0.2", "minp:0.2"),
            ("topp:0.9", "topp:0.9"),
            ("topp:1", "topp:1"),
            ("escape:16", "escape:16"),
            ("escape:8+floor:0.05", "floor:0.05+escape:8"),
            ("relax:16+temperature:0.75", "temperature:0.75+relax:16"),
        ] {
            let mode: ActMode = text.parse().unwrap();
            assert_eq!(mode.to_string(), canonical, "{text}");
            assert_eq!(canonical.parse::<ActMode>().unwrap(), mode);
        }
        for bad in [
            "", "model", "sample+floor:0.1", "argmax+escape:3", "floor", "floor:0", "floor:1", "floor:x",
            "temperature:0", "temperature:-1", "temperature:inf", "minp:1.5", "topp:0", "escape:0", "escape:1.5",
            "floor:0.1+floor:0.2", "escape:3+escape:4", "greedy:1", "relax:16", "relax:0+floor:0.1",
            "temperature:0.5+escape:4+relax:4", "relax:3+relax:4+floor:0.1", "argmax+relax:3",
        ] {
            assert!(bad.parse::<ActMode>().is_err(), "{bad:?} parsed");
        }
    }

    #[test]
    fn escape_switches_on_the_stall_count() {
        let mode: ActMode = "escape:4+floor:0.1".parse().unwrap();
        assert_eq!(mode.choose(3), Choose::Argmax);
        assert_eq!(mode.choose(4), Choose::Draw(Rule { floor: Some(0.1), ..Rule::default() }));
        assert_eq!(ActMode::Argmax.choose(1000), Choose::Argmax);
        assert_eq!(ActMode::SAMPLE.choose(0), Choose::Draw(Rule::default()));
        // relax: the rule until stalled, then the policy itself.
        let mode: ActMode = "temperature:0.75+relax:4".parse().unwrap();
        assert_eq!(mode.choose(3), Choose::Draw(Rule { temperature: Some(0.75), ..Rule::default() }));
        assert_eq!(mode.choose(4), Choose::Draw(Rule::default()));
        assert_eq!(mode.choose(1000), Choose::Draw(Rule::default()));
    }
}
