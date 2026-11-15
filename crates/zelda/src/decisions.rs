//! Per-room-visit decision statistics (`eval.json` `records[i]['visits'][j]['decisions']`, chain `RoomRecord`
//! `decisions`): how likely each chosen action was under the unmodified policy, what followed the unlikely ones,
//! the entropy of the distribution each action was drawn from (the act mode's), and the actions themselves (joint
//! indices), so runs from the same start can be compared for variety.
//!
//! A decision is *low* at a threshold when its probability under the policy is below it ([`LOW_PROBS`]: 1% and 5%;
//! a 1% one counts at both). It is *followed by a hit* when Link loses health on its own step or one of the next
//! `FOLLOW - 1` steps of the same visit ([`FOLLOW`] decisions in all, about half a second of play), and *followed by
//! a death* when the episode or run ends in death within them. Each decision counts at most once per kind. The
//! same is counted for every decision (`hit`, `death`): the base rate a low decision's is compared with.
//!
//! Decisions are also counted by Link's health when they were taken ([`DecisionStats::by_health`], half hearts), with
//! the hits their own step took: hits per decision at low and at normal health (round 25, `low` act modes).

use std::fmt::Write;

/// The probability thresholds of a low decision.
pub const LOW_PROBS: [f32; 2] = [0.01, 0.05];
/// Decisions after a low one (its own included) in which a hit or death is said to follow it.
pub const FOLLOW: usize = 4;

/// One decision as the actor reported it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionNote {
    /// The chosen action's log-prob under the policy.
    pub log_prob: f32,
    /// The entropy (nats) of the distribution the action was drawn from (0 for argmax).
    pub act_entropy: f32,
    /// The joint action index (`kind * 32 + direction * 4 + duration`).
    pub action: u8,
    /// Link's health (hearts) when the decision was taken; None where the env has none.
    pub health: Option<f32>,
}

/// A visit's decisions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecisionStats {
    pub decisions: u32,
    /// Per [`LOW_PROBS`] threshold: low decisions, and those followed by a hit and by a death.
    pub low: [u32; 2],
    pub low_hit: [u32; 2],
    pub low_death: [u32; 2],
    /// Decisions (any) followed by a hit, by a death.
    pub hit: u32,
    pub death: u32,
    /// Sum of -ln p of the chosen actions under the policy.
    pub surprisal: f64,
    /// Sum of the acting distributions' entropies.
    pub act_entropy: f64,
    /// The joint action of every decision, in order.
    pub actions: Vec<u8>,
    /// By Link's health at the decision in half hearts (index 5: 2.5 hearts): decisions, and those whose own step
    /// lost health. Only as long as the highest health seen.
    pub by_health: Vec<[u32; 2]>,
    /// The last [`FOLLOW`] steps' decisions: bit i below `LOW_PROBS[i]`, bit [`ANY`] a decision; most recent first.
    recent: [u8; FOLLOW],
    /// Whether a hit, a death was already counted for each of `recent`.
    hit_done: [bool; FOLLOW],
    dead: [bool; FOLLOW],
}

impl DecisionStats {
    /// Follows one step: the decision that chose it (None when the actor reported none) and whether the step lost
    /// health (`hit`) or ended in death.
    pub fn update(&mut self, note: Option<DecisionNote>, hit: bool, died: bool) {
        self.recent.rotate_right(1);
        self.hit_done.rotate_right(1);
        self.dead.rotate_right(1);
        self.recent[0] = 0;
        self.hit_done[0] = false;
        self.dead[0] = false;
        if let Some(note) = note {
            self.decisions += 1;
            self.surprisal -= note.log_prob as f64;
            self.act_entropy += note.act_entropy as f64;
            self.actions.push(note.action);
            if let Some(hearts) = note.health {
                let half = (hearts * 2.0).round().max(0.0) as usize;
                if self.by_health.len() <= half {
                    self.by_health.resize(half + 1, [0, 0]);
                }
                self.by_health[half][0] += 1;
                self.by_health[half][1] += hit as u32;
            }
            self.recent[0] |= 1 << ANY;
            for (i, &p) in LOW_PROBS.iter().enumerate() {
                if note.log_prob < p.ln() {
                    self.low[i] += 1;
                    self.recent[0] |= 1 << i;
                }
            }
        }
        let kinds = [
            (hit, &mut self.hit_done, &mut self.low_hit, &mut self.hit),
            (died, &mut self.dead, &mut self.low_death, &mut self.death),
        ];
        for (happened, done, counts, any) in kinds {
            if happened {
                for (&bits, done) in self.recent.iter().zip(done.iter_mut()) {
                    if !*done {
                        count(bits, counts, any);
                        *done = true;
                    }
                }
            }
        }
    }

    /// The actions as lowercase hex, two digits per decision.
    pub fn actions_hex(&self) -> String {
        let mut hex = String::with_capacity(self.actions.len() * 2);
        for a in &self.actions {
            write!(hex, "{a:02x}").expect("string write");
        }
        hex
    }
}

/// Bit of `DecisionStats::recent` for any decision.
const ANY: usize = LOW_PROBS.len();

fn count(bits: u8, into: &mut [u32; 2], any: &mut u32) {
    for (i, n) in into.iter_mut().enumerate() {
        if bits & (1 << i) != 0 {
            *n += 1;
        }
    }
    if bits & (1 << ANY) != 0 {
        *any += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(p: f32, action: u8) -> Option<DecisionNote> {
        Some(DecisionNote {
            log_prob: p.ln(),
            act_entropy: 0.5,
            action,
            health: None,
        })
    }

    #[test]
    fn decisions_by_health_count_their_own_steps_hits() {
        let mut s = DecisionStats::default();
        let at = |hearts: f32| Some(DecisionNote { health: Some(hearts), ..note(0.5, 0).unwrap() });
        s.update(at(3.0), false, false);
        s.update(at(3.0), true, false);
        s.update(at(2.5), false, false);
        s.update(None, true, false); // no decision: nothing by health
        s.update(at(0.5), true, true);
        assert_eq!(s.by_health, vec![[0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [1, 0], [2, 1]]);
    }

    #[test]
    fn low_decisions_and_what_follows_them() {
        let mut s = DecisionStats::default();
        s.update(note(0.005, 1), false, false); // below 1% and 5%
        s.update(note(0.03, 2), false, false); // below 5%
        s.update(note(0.9, 3), true, false); // a hit: both count once
        s.update(note(0.9, 4), true, false); // nothing low left in the window
        assert_eq!((s.low, s.low_hit, s.low_death), ([1, 2], [1, 2], [0, 0]));
        // A low decision whose window closes before the hit.
        s.update(note(0.02, 5), false, false);
        for _ in 1..FOLLOW {
            s.update(note(0.9, 6), false, false);
        }
        s.update(note(0.9, 7), true, true);
        assert_eq!((s.low, s.low_hit, s.low_death), ([1, 3], [1, 2], [0, 0]));
        // A death right after a low decision counts as a hit and a death.
        s.update(note(0.001, 8), false, false);
        s.update(None, true, true);
        assert_eq!((s.low, s.low_hit, s.low_death), ([2, 4], [2, 3], [1, 1]));
        // Every decision: 3 + 1 at the first two hits, 4 at the third (death too), 1 at the last.
        assert_eq!((s.hit, s.death), (3 + 1 + 4 + 1, 4 + 1));
        assert_eq!(s.decisions, 10);
        assert_eq!(s.actions.len(), 10);
        assert!((s.act_entropy - 5.0).abs() < 1e-9);
        assert_eq!(s.actions_hex(), "01020304050606060708");
    }
}
