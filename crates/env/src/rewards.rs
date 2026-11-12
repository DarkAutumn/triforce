//! Named reward/penalty outcomes, per-step clamping, terminal rewards and the scoring of one step.

use crate::config::SpecialistConfig;
use crate::critic::{Critique, GameplayCritic};
use crate::end_conditions::{EndConditions, MAX_STEPS, NO_DAMAGE, NO_ROOM_PROGRESS, StepView, classify_ending};
use crate::error::EnvError;
use zelda::game::GameView;

pub const REWARD_MINIMUM: f64 = 0.01;
pub const REWARD_TINY: f64 = 0.05;
pub const REWARD_SMALL: f64 = 0.25;
pub const REWARD_MEDIUM: f64 = 0.5;
pub const REWARD_LARGE: f64 = 0.75;
pub const REWARD_MAXIMUM: f64 = 1.0;
pub const TERMINAL_REWARD_MIN: f64 = -20.0;
pub const TERMINAL_FAILURE_PENALTY_VALUE: f64 = 20.0;
pub const TERMINAL_SUCCESS_REWARD_VALUE: f64 = 20.0;

pub const SUCCESS: &str = "success";
pub const HARD_FAILURE: &str = "hard-failure";
pub const SOFT_FAILURE: &str = "soft-failure";
/// Neither success nor failure: no terminal reward or penalty (`retreat`).
pub const NEUTRAL: &str = "neutral";
/// Terminal outcome classes (deny-by-default classification, `end_conditions::classify_ending`).
pub const OUTCOMES: [&str; 4] = [SUCCESS, HARD_FAILURE, SOFT_FAILURE, NEUTRAL];

pub const REWARD_TERMINAL_SUCCESS: &str = "reward-terminal-success";
pub const PENALTY_TERMINAL_FAILURE: &str = "penalty-terminal-failure";

/// Every reward and penalty term name the critic and terminal rewards can produce, rewards first (metrics).
pub const REWARD_TERMS: [&str; 55] = [
    "reward-gained-sword",
    "reward-gained-arrows",
    "reward-gained-bow",
    "reward-gained-candle",
    "reward-gained-whistle",
    "reward-gained-food",
    "reward-gained-potion",
    "reward-gained-magic_rod",
    "reward-gained-raft",
    "reward-gained-book",
    "reward-gained-ring",
    "reward-gained-ladder",
    "reward-gained-magic_key",
    "reward-gained-power_bracelet",
    "reward-gained-letter",
    "reward-gained-boomerang",
    "reward-gained-compass",
    "reward-gained-map",
    "reward-gained-rupees",
    "reward-gained-heart-container",
    "reward-gained-triforce",
    "reward-gained-bombs",
    "reward-gained-keys",
    "reward-gained-health",
    "reward-boss-hit",
    "reward-boss-stunned",
    "reward-beam-hit",
    "reward-beam-distance",
    "reward-hit",
    "reward-boomerang-stun",
    "reward-bomb-hit",
    "reward-armed-bomb-drop",
    "reward-opened-locked-door",
    "reward-opened-bomb-wall",
    "reward-pushed-block",
    "reward-new-location",
    "reward-revisit-location",
    "reward-moved-off-wallmaster",
    "reward-block-projectile",
    "reward-pbrs-movement",
    REWARD_TERMINAL_SUCCESS,
    "penalty-attack-miss",
    "penalty-attack-cave",
    "penalty-bomb-used",
    "penalty-boss-ate-bomb",
    "penalty-wrong-location",
    "penalty-fighting-wallmaster",
    "penalty-moved-onto-wallmaster",
    "penalty-quiet-enemies",
    "penalty-lost-health",
    "penalty-lost-beams",
    "penalty-room-stalling",
    "penalty-pbrs-movement",
    "penalty-frames",
    PENALTY_TERMINAL_FAILURE,
];

/// A named reward (value >= 0, name `reward*`) or penalty (value <= 0, name `penalty*`).
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub name: &'static str,
    pub value: f64,
    pub count: u32,
}

impl Outcome {
    pub fn reward(name: &'static str, value: f64) -> Outcome {
        debug_assert!(value >= 0.0 && name.starts_with("reward"), "{name} {value}");
        Outcome {
            name,
            value,
            count: 1,
        }
    }

    pub fn penalty(name: &'static str, value: f64) -> Outcome {
        debug_assert!(
            value <= 0.0 && name.starts_with("penalty"),
            "{name} {value}"
        );
        Outcome {
            name,
            value,
            count: 1,
        }
    }

    pub fn is_reward(&self) -> bool {
        self.name.starts_with("reward")
    }
}

/// A single step's rewards, in insertion order (`StepRewards`).
#[derive(Clone, Debug, Default)]
pub struct StepRewards {
    outcomes: Vec<Outcome>,
    /// End-condition name on the last step.
    pub ending: Option<String>,
    /// SUCCESS / HARD_FAILURE / SOFT_FAILURE / NEUTRAL on the last step.
    pub outcome: Option<&'static str>,
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.min(hi).max(lo)
}

impl StepRewards {
    pub fn iter(&self) -> impl Iterator<Item = &Outcome> {
        self.outcomes.iter()
    }

    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.outcomes.iter().any(|o| o.name == name)
    }

    pub fn get(&self, name: &str) -> Option<&Outcome> {
        self.outcomes.iter().find(|o| o.name == name)
    }

    /// (name, value) for every outcome this step (`to_dict`).
    pub fn terms(&self) -> impl Iterator<Item = (&'static str, f64)> + '_ {
        self.outcomes.iter().map(|o| (o.name, o.value))
    }

    /// The total, clamped to ±REWARD_MAXIMUM except for terminal rewards (Python `sum` order).
    pub fn value(&self) -> f64 {
        let total = self.outcomes.iter().fold(0.0, |acc, o| o.value + acc);
        if self.outcome == Some(SUCCESS) && self.contains(REWARD_TERMINAL_SUCCESS) {
            return clamp(total, -REWARD_MAXIMUM, TERMINAL_SUCCESS_REWARD_VALUE);
        }
        if matches!(self.outcome, Some(HARD_FAILURE | SOFT_FAILURE))
            && self.contains(PENALTY_TERMINAL_FAILURE)
        {
            return clamp(total, TERMINAL_REWARD_MIN, REWARD_MAXIMUM);
        }
        clamp(total, -REWARD_MAXIMUM, REWARD_MAXIMUM)
    }

    /// Adds an outcome; panics on a duplicate name (a critic bug).
    pub fn add(&mut self, outcome: Outcome) {
        assert!(
            !self.contains(outcome.name),
            "Duplicate outcome: {}",
            outcome.name
        );
        self.outcomes.push(outcome);
    }

    /// Adds `outcome` with its value scaled and clamped to ±REWARD_MAXIMUM (count resets to 1).
    pub fn add_scaled(&mut self, outcome: Outcome, scale: f64) {
        let value = clamp(outcome.value * scale, -REWARD_MAXIMUM, REWARD_MAXIMUM);
        self.add(Outcome {
            name: outcome.name,
            value,
            count: 1,
        });
    }

    /// Drops every reward (penalties stay).
    pub fn remove_rewards(&mut self) {
        self.outcomes.retain(|o| !o.is_reward());
    }

    /// Multiplies every outcome (rewards and penalties) by `factor` (`reward.shaping_scale`, outcome fine-tuning).
    pub fn scale_all(&mut self, factor: f64) {
        for o in &mut self.outcomes {
            o.value *= factor;
        }
    }

    /// Scales positive rewards by `factor` except those whose names start with an exempt prefix.
    pub fn scale_rewards(&mut self, factor: f64, exempt_prefixes: &[&str]) {
        for o in &mut self.outcomes {
            if o.is_reward() && !exempt_prefixes.iter().any(|p| o.name.starts_with(p)) {
                o.value *= factor;
            }
        }
    }
}

/// Success pays +20 × ((1 − w) + w × hp/max_hp), w = `reward.exit_health_weight` (default 0.5): health is handed to
/// the next specialist.
pub fn success_reward_value(health: f64, max_health: f64, exit_health_weight: f64) -> f64 {
    let frac = if max_health > 0.0 {
        (health / max_health).clamp(0.0, 1.0)
    } else {
        0.0
    };
    TERMINAL_SUCCESS_REWARD_VALUE * ((1.0 - exit_health_weight) + exit_health_weight * frac)
}

/// Records the ending/outcome and adds the terminal reward: success + health-scaled reward
/// (`success_reward_value`); hard failure drops the step's rewards (penalties kept) and adds −20; soft failure adds
/// −20; neutral and truncation add nothing.
pub fn apply_terminal_rewards(
    rewards: &mut StepRewards,
    ending: &str,
    outcome: &'static str,
    terminated: bool,
    health: f64,
    max_health: f64,
    exit_health_weight: f64,
) {
    assert!(OUTCOMES.contains(&outcome), "Unknown outcome {outcome:?}");
    rewards.ending = Some(ending.to_owned());
    rewards.outcome = Some(outcome);
    if !terminated {
        return;
    }
    if outcome == SUCCESS {
        rewards.add(Outcome::reward(
            REWARD_TERMINAL_SUCCESS,
            success_reward_value(health, max_health, exit_health_weight),
        ));
    } else if outcome != NEUTRAL {
        if outcome == HARD_FAILURE {
            rewards.remove_rewards();
        }
        rewards.add(Outcome::penalty(
            PENALTY_TERMINAL_FAILURE,
            -TERMINAL_FAILURE_PENALTY_VALUE,
        ));
    }
}

/// What one step of a specialist's episode earned and whether it ended the episode.
#[derive(Clone, Debug)]
pub struct Scored {
    pub rewards: StepRewards,
    /// The end condition that fired (`end_conditions`), or `max-steps`.
    pub ending: Option<String>,
    /// The ending's class ([`OUTCOMES`]).
    pub outcome: Option<&'static str>,
    pub terminated: bool,
    pub truncated: bool,
}

/// Scores one step of `config`'s specialist: the critic's terms (scaled by `scale`, the shaping scale), the
/// frame penalty, the end conditions (a `no-room-progress`, `no-damage` or `max-steps` ending truncates) and
/// the terminal reward.  `c.quiet_steps` is the linger tail after the step; `steps` the actions of the episode
/// including this one; `state` the state the step ended in.  Training (`SpecialistEnv`) and a chain's reward
/// recording (`RewardTrack`) both score with this.
pub fn score_step(
    config: &SpecialistConfig,
    critic: &mut GameplayCritic,
    end_conditions: &mut EndConditions,
    c: &Critique,
    scale: f64,
    steps: u64,
    state: &GameView,
) -> Result<Scored, EnvError> {
    let mut rewards = StepRewards::default();
    critic.critique(c, &mut rewards);
    if scale != 1.0 {
        rewards.scale_all(scale);
    }
    if config.reward.frame_penalty > 0.0 {
        rewards.add(Outcome::penalty(
            "penalty-frames",
            -config.reward.frame_penalty * c.change.frames as f64,
        ));
    }
    let mut ending = end_conditions.check(&StepView {
        change: c.change,
        prev_objective: c.prev_objective,
        map: c.map,
        linger_tail_steps: c.quiet_steps,
    });
    let mut truncated = matches!(ending.as_deref(), Some(NO_ROOM_PROGRESS | NO_DAMAGE));
    if ending.is_none() && steps >= config.max_steps as u64 {
        ending = Some(MAX_STEPS.to_owned());
        truncated = true;
    }
    let terminated = ending.is_some() && !truncated;
    let mut outcome = None;
    if let Some(ending) = &ending {
        let class = classify_ending(ending, &config.success, &config.endings)?;
        apply_terminal_rewards(
            &mut rewards,
            ending,
            class,
            terminated,
            state.health(),
            state.max_health() as f64,
            config.reward.exit_health_weight,
        );
        outcome = Some(class);
    }
    Ok(Scored {
        rewards,
        ending,
        outcome,
        terminated,
        truncated,
    })
}
