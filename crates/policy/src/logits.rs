//! Acting from given head logits: what [`Policy::act`] does after its trunk, for a network that runs elsewhere
//! (an experimental torch module served from Python, docs/lab.md). Given the same logits, the masks, the forced-kind
//! rule, the keyed noise, the act-mode rules and the outputs are [`Policy::act`]'s, bit for bit.
//!
//! The logits are laid out like the action mask: the kind head's `kinds` values, then every kind's direction head
//! (`kinds × directions`, row `k`: the direction given kind `k`), then every kind's duration head (`kinds ×
//! durations`). A network whose direction and duration heads don't depend on the kind repeats its rows.

use super::{
    Acting, Choose, LEAD_KINDS, MASKED_LOGIT, MAX_HEAD, Output, Pick, PolicyError, Rule, act, err, finite_output, head,
    masked_log_softmax, noise, sub_head, sub_log_probs,
};

/// The factored action heads' sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Heads {
    pub kinds: usize,
    pub directions: usize,
    pub durations: usize,
}

/// Working memory of [`Heads::act`] (no allocation per call).
#[derive(Clone, Debug)]
pub struct LogitScratch {
    joint: Vec<f32>,
    acting: Vec<f64>,
    order: Vec<usize>,
    joint_noise: Vec<f32>,
}

impl Heads {
    /// Values per row of the logits (and of the action mask).
    pub fn size(&self) -> usize {
        self.kinds * (1 + self.directions + self.durations)
    }

    /// Joint actions (`Policy::joint_size`).
    pub fn joint_size(&self) -> usize {
        self.kinds * self.directions * self.durations
    }

    pub fn scratch(&self) -> LogitScratch {
        let n = self.joint_size();
        LogitScratch {
            joint: vec![0.0; n],
            acting: vec![0.0; n],
            order: vec![0; n],
            joint_noise: vec![0.0; n],
        }
    }

    fn joint_index(&self, action: [usize; 3]) -> usize {
        (action[0] * self.directions + action[1]) * self.durations + action[2]
    }

    fn joint_action(&self, index: usize) -> [usize; 3] {
        let per_kind = self.directions * self.durations;
        [index / per_kind, (index % per_kind) / self.durations, index % self.durations]
    }

    fn noise_size(&self) -> usize {
        self.kinds + self.directions + self.durations
    }

    /// [`Policy::act`] on `logits` (this layout) with value `value` under the flat action `mask`: the action of
    /// decision `step` of job `job` the way `choose` says, or `given` scored.
    #[allow(clippy::too_many_arguments)]
    pub fn act(
        &self,
        logits: &[f32],
        value: f32,
        mask: &[bool],
        choose: Choose,
        job: i64,
        step: i64,
        given: Option<[usize; 3]>,
        s: &mut LogitScratch,
    ) -> Result<(Output, Acting), PolicyError> {
        let (kinds, dirs, durs) = (self.kinds, self.directions, self.durations);
        if logits.len() != self.size() || mask.len() != self.size() {
            return err(format!(
                "logits ({}) and mask ({}) need {} values each",
                logits.len(),
                mask.len(),
                self.size()
            ));
        }
        if kinds.max(dirs).max(durs) > MAX_HEAD {
            return err(format!("a head larger than {MAX_HEAD}"));
        }
        if let Some([k, d, u]) = given
            && (k >= kinds || d >= dirs || u >= durs)
        {
            return err(format!("action [{k}, {d}, {u}] is outside the {kinds}/{dirs}/{durs} heads"));
        }
        let legal_kinds = match mask[..kinds].iter().filter(|&&m| m).count() {
            0 => return err("empty kind mask"),
            n => n,
        };
        self.joint(logits, mask, legal_kinds, &mut s.joint)?;
        let out = match (choose, given) {
            (Choose::Argmax, None) => (self.heads(logits, value, mask, Pick::Argmax)?, Acting::default()),
            (Choose::Argmax, Some(action)) => (self.heads(logits, value, mask, Pick::Given(action))?, Acting::default()),
            (Choose::Draw(rule), Some(action)) => {
                let out = self.heads(logits, value, mask, Pick::Given(action))?;
                let acting = if rule.is_plain() {
                    Acting {
                        log_prob: out.log_prob,
                        entropy: s.joint.iter().filter(|v| v.is_finite()).map(|&v| -v.exp() * v).sum(),
                    }
                } else {
                    rule.probabilities(&s.joint, &mut s.acting, &mut s.order);
                    Acting {
                        log_prob: s.acting[self.joint_index(action)].ln() as f32,
                        entropy: act::entropy(&s.acting) as f32,
                    }
                };
                (out, acting)
            }
            (Choose::Draw(rule), None) if rule.is_plain() => {
                let mut noise = [0f32; 3 * MAX_HEAD];
                let noise = &mut noise[..self.noise_size()];
                noise::keyed_noise(job, step, noise);
                let out = self.heads(logits, value, mask, Pick::Sample(noise))?;
                let entropy: f32 = s.joint.iter().filter(|v| v.is_finite()).map(|&v| -v.exp() * v).sum();
                (out, Acting { log_prob: out.log_prob, entropy })
            }
            (Choose::Draw(rule), None) => {
                rule.probabilities(&s.joint, &mut s.acting, &mut s.order);
                noise::gumbel(act::joint_noise_key(job, step), &mut s.joint_noise);
                let index = Rule::draw(&s.acting, &s.joint_noise);
                let acting = Acting {
                    log_prob: s.acting[index].ln() as f32,
                    entropy: act::entropy(&s.acting) as f32,
                };
                (self.heads(logits, value, mask, Pick::Given(self.joint_action(index)))?, acting)
            }
        };
        Ok(out)
    }

    /// `Policy::heads` on given logits.
    fn heads(&self, logits: &[f32], value: f32, mask: &[bool], pick: Pick) -> Result<Output, PolicyError> {
        let (kinds, dirs, durs) = (self.kinds, self.directions, self.durations);
        let kind_mask = &mask[..kinds];
        let mut out = Output {
            action: [0; 3],
            log_prob: 0.0,
            entropy: 0.0,
            value,
            head_log_probs: [[0.0; MAX_HEAD]; 3],
            head_applies: [false; 3],
            head_entropy: [0.0; 3],
            head_log_prob: [0.0; 3],
        };
        let lead = kinds.min(LEAD_KINDS);
        let mut kind_noise = [0f32; MAX_HEAD];
        let kind_pick = match pick {
            Pick::Sample(noise) => {
                kind_noise[..lead].copy_from_slice(&noise[..lead]);
                kind_noise[lead..kinds].copy_from_slice(&noise[lead + dirs + durs..]);
                Pick::Sample(&kind_noise[..kinds])
            }
            other => other,
        };
        let kind = head(&logits[..kinds], kind_mask, kind_pick, 0, &mut out, 0);
        out.head_applies[0] = kind_mask.iter().filter(|&&m| m).count() > 1;
        if !out.head_applies[0] {
            out.head_entropy[0] = 0.0;
            out.head_log_prob[0] = 0.0;
        }
        let dir_at = kinds + kind * dirs;
        sub_head(&logits[dir_at..dir_at + dirs], &mask[dir_at..dir_at + dirs], pick, lead, &mut out, 1);
        let dur_at = kinds + kinds * dirs + kind * durs;
        sub_head(&logits[dur_at..dur_at + durs], &mask[dur_at..dur_at + durs], pick, lead + dirs, &mut out, 2);
        out.action[0] = kind;
        out.log_prob = (out.head_log_prob[0] + out.head_log_prob[1]) + out.head_log_prob[2];
        out.entropy = (out.head_entropy[0] + out.head_entropy[1]) + out.head_entropy[2];
        finite_output(out.value, out.log_prob)?;
        Ok(out)
    }

    /// `Policy::joint` on given logits.
    fn joint(&self, logits: &[f32], mask: &[bool], legal_kinds: usize, out: &mut [f32]) -> Result<(), PolicyError> {
        let (kinds, dirs, durs) = (self.kinds, self.directions, self.durations);
        let kind_mask = &mask[..kinds];
        out.fill(f32::NEG_INFINITY);
        let mut kind_lp = [0f32; MAX_HEAD];
        masked_log_softmax(&logits[..kinds], kind_mask, MASKED_LOGIT, &mut kind_lp[..kinds]);
        let dur_base = kinds + kinds * dirs;
        for kind in (0..kinds).filter(|&k| kind_mask[k]) {
            let dir_at = kinds + kind * dirs;
            let dir_mask = &mask[dir_at..dir_at + dirs];
            let (dir_lp, dir_applies) = sub_log_probs(&logits[dir_at..dir_at + dirs], dir_mask);
            let dur_at = dur_base + kind * durs;
            let dur_mask = &mask[dur_at..dur_at + durs];
            let (dur_lp, dur_applies) = sub_log_probs(&logits[dur_at..dur_at + durs], dur_mask);
            let lk = if legal_kinds > 1 { kind_lp[kind] } else { 0.0 };
            for d in 0..dirs {
                let ld = match dir_applies {
                    true if dir_mask[d] => dir_lp[d],
                    false if d == 0 => 0.0,
                    _ => continue,
                };
                for u in 0..durs {
                    let lu = match dur_applies {
                        true if dur_mask[u] => dur_lp[u],
                        false if u == 0 => 0.0,
                        _ => continue,
                    };
                    out[(kind * dirs + d) * durs + u] = (lk + ld) + lu;
                }
            }
        }
        if out.iter().any(|v| v.is_nan()) {
            return err("non-finite policy output");
        }
        Ok(())
    }
}
