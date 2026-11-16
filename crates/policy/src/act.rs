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
//! of `escape` and `relax`; `sample` and `argmax` stand alone. That is a [`Style`]. A mode is a style, optionally
//! followed by `+low:H:STYLE`, the style it plays instead while Link holds H hearts or fewer (`low:2`; halves allowed)
//! or H percent of his maximum or less (`low:40%`), read before every decision: `temperature:0.5+relax:16+low:2:
//! temperature:0.25+relax:16`. The low part comes last (everything after `low:H:` is its style) and has its own stall
//! rule. Either style may end with one foresight part, `+guard:K` or `+plan:K:S[:R]` ([`Foresight`]: the env tries
//! the decision's action, and when it would lose health the next of the style's K best that would not).
//! [`ActMode::to_string`] is the canonical spelling (rules in the table's order, the foresight part last). A rule's
//! draw is Gumbel-argmax over the joint actions with noise keyed by (job, decision) ([`joint_noise_key`]), so a mode
//! plays the same for the same seed wherever it runs.

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

/// One way of acting: what a mode does at normal health, or its low-health part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Style {
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

/// Where low health starts: Link holds this many hearts or fewer, or this share (percent) of his maximum or less.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Threshold {
    Hearts(f32),
    Percent(f32),
}

/// Foresight before a decision (round 25): the env tries candidates in turn, the decision's own action first, then
/// the style's next best (its ranking: argmax by the policy's joint log-prob, a draw by ln q + the decision's own
/// Gumbel noise), at most `candidates` in all, each against `futures` futures, and takes the first that loses no
/// health ([`Foresight::refuses`]); when every candidate would, the decision's own action. A trial executes the action
/// and returns to exactly where the env stood: nothing of it reaches the game, the run's counters or a recording.
///
/// - `guard:K` (`futures` 0): each candidate is tried once against the game's own future (its random numbers as they
///   stand). The run is a valid sequence of button presses, chosen with knowledge a player doesn't have.
/// - `plan:K:S[:R]` (`futures` S ≥ 1): the game's random numbers are redrawn before every trial (a natural RNG state of
///   a phase keyed by the job, the decision and the future, the same S futures for every candidate), and a candidate
///   is refused when it loses health in more than the share R of them (default 0: in any). Planning with a model of
///   the game, not peeking at its future.
///
/// K = 1 tries only the decision's own action: play is unchanged, the trials are only counted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Foresight {
    pub candidates: u32,
    pub futures: u32,
    pub share: f32,
}

impl Foresight {
    /// Trials of one candidate: one against the game's own future, else one per redrawn future.
    pub fn trials(&self) -> u32 {
        self.futures.max(1)
    }

    /// Whether a candidate that lost health in `losses` of its [`Foresight::trials`] is refused.
    pub fn refuses(&self, losses: u32) -> bool {
        losses as f64 > self.share as f64 * self.trials() as f64
    }

    /// Whether `losses` of `done` trials already decide [`Foresight::refuses`] (the rest can't change it).
    pub fn decided(&self, losses: u32, done: u32) -> bool {
        self.refuses(losses) || !self.refuses(losses + (self.trials() - done))
    }
}

/// The low-health part of a mode (`+low:H:STYLE`): `style` (and its foresight) while Link's health is at or under
/// `at`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Low {
    pub at: Threshold,
    pub style: Style,
    pub foresight: Option<Foresight>,
}

/// How a policy acts outside training: `style` with its `foresight`, and `low`'s instead while Link is at low health
/// (docs/rust-actor.md "Act modes").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActMode {
    pub style: Style,
    pub foresight: Option<Foresight>,
    pub low: Option<Low>,
}

/// How one decision is taken: by argmax, or drawn by a rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Choose {
    Argmax,
    Draw(Rule),
}

impl Style {
    /// How the decision after `stall` decisions in a row without progress is taken.
    pub fn choose(&self, stall: u32) -> Choose {
        match *self {
            Style::Argmax => Choose::Argmax,
            Style::Sample(rule) => Choose::Draw(rule),
            Style::Escape { after, rule } => {
                if stall >= after {
                    Choose::Draw(rule)
                } else {
                    Choose::Argmax
                }
            }
            Style::Relax { after, rule } => Choose::Draw(if stall >= after { Rule::default() } else { rule }),
        }
    }
}

impl Threshold {
    /// Whether `hearts` of `max` is at or under the threshold.
    pub fn holds(&self, hearts: f32, max: f32) -> bool {
        match *self {
            Threshold::Hearts(h) => hearts <= h,
            // In hundredths: exact for half hearts and whole percents.
            Threshold::Percent(p) => hearts as f64 * 100.0 <= p as f64 * max as f64,
        }
    }
}

impl ActMode {
    pub const SAMPLE: ActMode = ActMode {
        style: Style::Sample(Rule {
            temperature: None,
            floor: None,
            minp: None,
            topp: None,
        }),
        foresight: None,
        low: None,
    };
    pub const ARGMAX: ActMode = ActMode {
        style: Style::Argmax,
        foresight: None,
        low: None,
    };

    /// The style and foresight for Link's `health` (hearts held, maximum; None where the env has none: never low).
    fn part(&self, health: Option<(f32, f32)>) -> (Style, Option<Foresight>) {
        match (self.low, health) {
            (Some(low), Some((hearts, max))) if low.at.holds(hearts, max) => (low.style, low.foresight),
            _ => (self.style, self.foresight),
        }
    }

    /// The style for Link's `health` (hearts held, maximum; None where the env has none: never low).
    pub fn style(&self, health: Option<(f32, f32)>) -> Style {
        self.part(health).0
    }

    /// The foresight of a decision at Link's `health` (None: the decision is taken as the style chooses it).
    pub fn foresight(&self, health: Option<(f32, f32)>) -> Option<Foresight> {
        self.part(health).1
    }

    /// How the decision after `stall` decisions in a row without progress, with Link's `health` (hearts held,
    /// maximum), is taken.
    pub fn choose(&self, stall: u32, health: Option<(f32, f32)>) -> Choose {
        self.style(health).choose(stall)
    }
}

impl Default for ActMode {
    fn default() -> Self {
        ActMode::SAMPLE
    }
}

impl fmt::Display for Style {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        match self {
            Style::Argmax => return f.write_str("argmax"),
            Style::Sample(rule) if rule.is_plain() => return f.write_str("sample"),
            Style::Sample(rule) => rule.fmt_parts(&mut parts),
            Style::Escape { after, rule } => {
                rule.fmt_parts(&mut parts);
                parts.push(format!("escape:{after}"));
            }
            Style::Relax { after, rule } => {
                rule.fmt_parts(&mut parts);
                parts.push(format!("relax:{after}"));
            }
        }
        f.write_str(&parts.join("+"))
    }
}

impl fmt::Display for Threshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Threshold::Hearts(h) => write!(f, "{h}"),
            Threshold::Percent(p) => write!(f, "{p}%"),
        }
    }
}

impl fmt::Display for Foresight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (k, s, r) = (self.candidates, self.futures, self.share);
        if s == 0 {
            write!(f, "guard:{k}")
        } else if r == 0.0 {
            write!(f, "plan:{k}:{s}")
        } else {
            write!(f, "plan:{k}:{s}:{r}")
        }
    }
}

impl fmt::Display for ActMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.style)?;
        if let Some(foresight) = &self.foresight {
            write!(f, "+{foresight}")?;
        }
        if let Some(low) = &self.low {
            write!(f, "+low:{}:{}", low.at, low.style)?;
            if let Some(foresight) = &low.foresight {
                write!(f, "+{foresight}")?;
            }
        }
        Ok(())
    }
}

impl FromStr for ActMode {
    type Err = String;

    /// `STYLE` or `STYLE+low:H:STYLE` (the low part last: everything after `low:H:` is its style); either style may
    /// carry one foresight part, `+guard:K` or `+plan:K:S[:R]`.
    fn from_str(text: &str) -> Result<ActMode, String> {
        let text = text.trim();
        let (normal, low) = match text.find("low:") {
            None => (text, None),
            Some(0) => return Err(format!("act mode {text:?}: low:H:MODE follows the mode for normal health")),
            Some(i) if !text[..i].ends_with('+') => {
                return Err(format!("act mode {text:?}: low:H:MODE must be its own '+' part"));
            }
            Some(i) => (&text[..i - 1], Some(&text[i + "low:".len()..])),
        };
        let (style, foresight) = parse_foresight(text, normal)?;
        let low = match low {
            None => None,
            Some(rest) => {
                let (at, style) = rest
                    .split_once(':')
                    .ok_or_else(|| format!("act mode {text:?}: low needs a threshold and a mode (low:H:MODE)"))?;
                if style.contains("low:") {
                    return Err(format!("act mode {text:?}: low given twice"));
                }
                let (style, foresight) = parse_foresight(text, style)?;
                Some(Low {
                    at: parse_threshold(text, at)?,
                    style,
                    foresight,
                })
            }
        };
        Ok(ActMode { style, foresight, low })
    }
}

/// A style with an optional foresight part (`guard:K`, `plan:K:S[:R]`; K ≥ 1 candidates, S ≥ 1 futures, 0 ≤ R < 1);
/// `text` is the whole mode.
fn parse_foresight(text: &str, part: &str) -> Result<(Style, Option<Foresight>), String> {
    let mut foresight = None;
    let mut rest = Vec::new();
    for item in part.split('+') {
        let (name, args) = match item.split_once(':') {
            Some((name @ ("guard" | "plan"), args)) => (name, args),
            _ => {
                rest.push(item);
                continue;
            }
        };
        if foresight.is_some() {
            return Err(format!("act mode {text:?}: one foresight part (guard or plan) per mode"));
        }
        let whole = |what: &str, v: &str| {
            v.parse::<u32>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| format!("act mode {text:?}: {name} needs a whole number of {what}, 1 or more, got {v:?}"))
        };
        let args: Vec<&str> = args.split(':').collect();
        foresight = Some(match (name, args.as_slice()) {
            ("guard", [k]) => Foresight {
                candidates: whole("candidates", k)?,
                futures: 0,
                share: 0.0,
            },
            ("plan", [k, s, r @ ..]) if r.len() <= 1 => {
                let share = match r {
                    [] => 0.0,
                    [r] => r
                        .parse::<f32>()
                        .ok()
                        .filter(|r| (0.0..1.0).contains(r))
                        .ok_or_else(|| format!("act mode {text:?}: plan's share must be in [0, 1), got {r:?}"))?,
                    _ => unreachable!(),
                };
                Foresight {
                    candidates: whole("candidates", k)?,
                    futures: whole("futures", s)?,
                    share,
                }
            }
            _ => return Err(format!("act mode {text:?}: {item:?} is not guard:K or plan:K:S[:R]")),
        });
    }
    if foresight.is_some() && rest.is_empty() {
        return Err(format!("act mode {text:?}: foresight needs a mode to act by"));
    }
    Ok((parse_style(text, &rest.join("+"))?, foresight))
}

/// `H` (hearts: a positive multiple of a half) or `P%` (a share of the maximum, 0 < P < 100).
fn parse_threshold(text: &str, at: &str) -> Result<Threshold, String> {
    let bad = || {
        format!(
            "act mode {text:?}: low's threshold {at:?} is neither hearts (a positive multiple of 0.5) nor a percent \
             of the maximum (0 < P < 100, e.g. 40%)"
        )
    };
    if let Some(p) = at.strip_suffix('%') {
        let p: f32 = p.parse().map_err(|_| bad())?;
        return if p.is_finite() && p > 0.0 && p < 100.0 {
            Ok(Threshold::Percent(p))
        } else {
            Err(bad())
        };
    }
    let h: f32 = at.parse().map_err(|_| bad())?;
    if h.is_finite() && h > 0.0 && (h * 2.0).fract() == 0.0 {
        Ok(Threshold::Hearts(h))
    } else {
        Err(bad())
    }
}

/// One style's spelling `part` (`sample`, `argmax`, or rules joined by `+`); `text` is the whole mode, for messages.
fn parse_style(text: &str, part: &str) -> Result<Style, String> {
    match part {
        "sample" => return Ok(Style::Sample(Rule::default())),
        "argmax" => return Ok(Style::Argmax),
        "" if text.is_empty() => return Err("empty act mode".to_owned()),
        "" => return Err(format!("act mode {text:?}: an empty mode")),
        _ => {}
    }
    let mut rule = Rule::default();
    // The stall rule: ("escape" | "relax", N).
    let mut stall: Option<(&str, u32)> = None;
    for part in part.split('+') {
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
                     topp:P, escape:N, relax:N, guard:K, plan:K:S[:R], low:H:MODE)"
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
        Some(("escape", after)) => Style::Escape { after, rule },
        Some((_, after)) if rule.is_plain() => {
            return Err(format!(
                "act mode {text:?}: relax:{after} needs another rule (alone it is sample at every decision)"
            ));
        }
        Some((_, after)) => Style::Relax { after, rule },
        None => Style::Sample(rule),
    })
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
            ("relax:16+temperature:0.5+low:2:temperature:0.25+relax:16", "temperature:0.5+relax:16+low:2:temperature:0.25+relax:16"),
            ("temperature:0.5+relax:16+low:1.5:relax:8+floor:0.1", "temperature:0.5+relax:16+low:1.5:floor:0.1+relax:8"),
            ("argmax+low:40%:escape:16", "argmax+low:40%:escape:16"),
            ("sample+low:3:argmax", "sample+low:3:argmax"),
            ("escape:16+low:25%:sample", "escape:16+low:25%:sample"),
            ("guard:4+temperature:0.5+relax:16", "temperature:0.5+relax:16+guard:4"),
            ("temperature:0.5+relax:16+low:2:guard:8+temperature:0.5+relax:16",
             "temperature:0.5+relax:16+low:2:temperature:0.5+relax:16+guard:8"),
            ("argmax+guard:2", "argmax+guard:2"),
            ("argmax+guard:1", "argmax+guard:1"),
            ("sample+guard:3+low:1:argmax+guard:5", "sample+guard:3+low:1:argmax+guard:5"),
            ("plan:8:4+temperature:0.5+relax:16", "temperature:0.5+relax:16+plan:8:4"),
            ("temperature:0.5+relax:16+plan:8:4:0", "temperature:0.5+relax:16+plan:8:4"),
            ("temperature:0.5+relax:16+plan:8:4:0.50", "temperature:0.5+relax:16+plan:8:4:0.5"),
            ("temperature:0.5+relax:16+low:2:temperature:0.5+relax:16+plan:8:1",
             "temperature:0.5+relax:16+low:2:temperature:0.5+relax:16+plan:8:1"),
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
            "low:2:argmax", "argmax+low:2", "argmax+low:2:", "argmax+low:0:sample", "argmax+low:-1:sample",
            "argmax+low:1.25:sample", "argmax+low:100%:sample", "argmax+low:0%:sample", "argmax+low:x:sample",
            "argmax+low:2:relax:4", "argmax+low:2:sample+low:1:argmax", "argmax+low:2:argmax+low:1:sample",
            "argmaxlow:2:sample", "+low:2:sample", "argmax+low:inf:sample",
            "guard:4", "argmax+guard:0", "argmax+guard:x", "argmax+guard:2+guard:3", "argmax+guard:2:1",
            "argmax+low:2:guard:4", "argmax+guard:4+plan:4:2", "argmax+plan:4", "argmax+plan:4:0",
            "argmax+plan:0:4", "argmax+plan:4:2:1", "argmax+plan:4:2:-0.1", "argmax+plan:4:2:0.5:1", "plan:4:2",
            "argmax+plan:4:2:x",
        ] {
            assert!(bad.parse::<ActMode>().is_err(), "{bad:?} parsed");
        }
    }

    #[test]
    fn escape_switches_on_the_stall_count() {
        let mode: ActMode = "escape:4+floor:0.1".parse().unwrap();
        assert_eq!(mode.choose(3, None), Choose::Argmax);
        assert_eq!(mode.choose(4, None), Choose::Draw(Rule { floor: Some(0.1), ..Rule::default() }));
        assert_eq!(ActMode::ARGMAX.choose(1000, None), Choose::Argmax);
        assert_eq!(ActMode::SAMPLE.choose(0, None), Choose::Draw(Rule::default()));
        // relax: the rule until stalled, then the policy itself.
        let mode: ActMode = "temperature:0.75+relax:4".parse().unwrap();
        assert_eq!(mode.choose(3, None), Choose::Draw(Rule { temperature: Some(0.75), ..Rule::default() }));
        assert_eq!(mode.choose(4, None), Choose::Draw(Rule::default()));
        assert_eq!(mode.choose(1000, None), Choose::Draw(Rule::default()));
    }

    #[test]
    fn foresight_follows_the_health_part_and_refuses_by_its_share() {
        let mode: ActMode = "temperature:0.5+relax:16+low:2:temperature:0.5+relax:16+guard:8".parse().unwrap();
        let guard = Foresight { candidates: 8, futures: 0, share: 0.0 };
        assert_eq!(mode.foresight(Some((2.5, 5.0))), None);
        assert_eq!(mode.foresight(Some((2.0, 5.0))), Some(guard));
        assert_eq!(mode.foresight(None), None);
        assert_eq!(mode.choose(0, Some((2.0, 5.0))), mode.choose(0, Some((3.0, 5.0))));
        let plan = "argmax+plan:3:4:0.5".parse::<ActMode>().unwrap().foresight(None).unwrap();
        assert_eq!(plan, Foresight { candidates: 3, futures: 4, share: 0.5 });
        // The game's own future: one trial, refused on a loss.
        assert_eq!(guard.trials(), 1);
        assert!(!guard.refuses(0) && guard.refuses(1));
        // Four redrawn futures, refused when more than half lose: 3 of 4.
        assert_eq!(plan.trials(), 4);
        assert_eq!((0..=4).map(|l| plan.refuses(l)).collect::<Vec<_>>(), [false, false, false, true, true]);
        // Decided early: 3 losses refuse whatever follows; 1 loss in 3 trials can reach only 2 of 4.
        assert!(plan.decided(3, 3) && plan.decided(1, 3) && !plan.decided(2, 3) && !plan.decided(0, 1));
        // "Any" (share 0): the first loss decides.
        let any = Foresight { share: 0.0, ..plan };
        assert!(any.decided(1, 1) && !any.decided(0, 3) && any.decided(0, 4));
    }

    #[test]
    fn low_switches_on_the_health_both_ways_and_keeps_each_stall_rule() {
        let t = |v: f32| Choose::Draw(Rule { temperature: Some(v), ..Rule::default() });
        let mode: ActMode = "temperature:0.5+relax:16+low:2:temperature:0.25+relax:8".parse().unwrap();
        // Hearts as Link loses them, then as he heals: the low style from 2 hearts down, the normal one above.
        let max = 5.0;
        for (hearts, want) in [(5.0, t(0.5)), (2.5, t(0.5)), (2.0, t(0.25)), (0.5, t(0.25)), (2.0, t(0.25)), (2.5, t(0.5))]
        {
            assert_eq!(mode.choose(0, Some((hearts, max))), want, "{hearts} hearts");
        }
        // Each state keeps its own stall rule: relax:16 above the threshold, relax:8 at or under it.
        assert_eq!(mode.choose(8, Some((3.0, max))), t(0.5));
        assert_eq!(mode.choose(16, Some((3.0, max))), Choose::Draw(Rule::default()));
        assert_eq!(mode.choose(7, Some((1.0, max))), t(0.25));
        assert_eq!(mode.choose(8, Some((1.0, max))), Choose::Draw(Rule::default()));
        // An env without health is never low.
        assert_eq!(mode.choose(0, None), t(0.5));
        // A share of the maximum: 40% of 5 hearts is 2, of 8 is 3.2.
        let mode: ActMode = "temperature:0.5+relax:16+low:40%:escape:16".parse().unwrap();
        assert_eq!(mode.choose(0, Some((2.0, 5.0))), Choose::Argmax);
        assert_eq!(mode.choose(0, Some((2.5, 5.0))), t(0.5));
        assert_eq!(mode.choose(0, Some((3.0, 8.0))), Choose::Argmax);
        assert_eq!(mode.choose(0, Some((3.5, 8.0))), t(0.5));
        assert_eq!(mode.choose(16, Some((1.0, 8.0))), Choose::Draw(Rule::default()));
        // 25% of 4 and of 8 hearts, exactly on the line.
        let mode: ActMode = "sample+low:25%:argmax".parse().unwrap();
        assert_eq!(mode.choose(0, Some((1.0, 4.0))), Choose::Argmax);
        assert_eq!(mode.choose(0, Some((2.0, 8.0))), Choose::Argmax);
        assert_eq!(mode.choose(0, Some((2.5, 8.0))), Choose::Draw(Rule::default()));
    }
}
