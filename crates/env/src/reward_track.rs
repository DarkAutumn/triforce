//! A specialist's reward bookkeeping outside its env: what training would pay for the steps a chain plays.
//!
//! [`RewardTrack`] holds the trackers `SpecialistEnv` keeps per episode (critic, end conditions, linger, hearts,
//! the attached objective) and scores every step with the same [`score_step`], in `SpecialistEnv::step_as`'s
//! order.  A chain makes one per acting specialist and restarts it when that specialist's own end conditions
//! fire, so a run is cut into the episodes training would see.

use crate::config::{ConfigError, SpecialistConfig};
use crate::critic::{Critique, GameplayCritic};
use crate::end_conditions::EndConditions;
use crate::error::EnvError;
use crate::objective::{Attached, attach_objective, needs_wavefront};
use crate::rewards::{Scored, score_step};
use std::sync::Arc;
use zelda::game::GameView;
use zelda::hearts::{HeartTracker, HeartView};
use zelda::linger::{LingerState, LingerStep, LingerTracker, linger_wavefront};
use zelda::objectives::Objective;
use zelda::{Change, GameMap};

/// One specialist's episode as its training env would score it.
pub struct RewardTrack {
    config: Arc<SpecialistConfig>,
    critic: GameplayCritic,
    end_conditions: EndConditions,
    linger: LingerTracker,
    hearts: HeartTracker,
    attached: Attached,
    steps: u64,
    ended: bool,
}

fn attach(config: &SpecialistConfig, state: &GameView, objective: &Objective) -> Attached {
    attach_objective(objective.clone(), state, &config.reward, needs_wavefront(config))
}

impl RewardTrack {
    /// An episode of `config`'s specialist starting at `state`, whose route objective is `objective`.
    pub fn new(config: Arc<SpecialistConfig>, state: &GameView, objective: &Objective) -> Result<Self, ConfigError> {
        let mut track = RewardTrack {
            critic: GameplayCritic::for_config(&config)?,
            end_conditions: EndConditions::new(
                &config.rooms,
                &config.success,
                config.one_room,
                config.no_progress_steps.map(|n| n as u32),
                config.no_damage_steps.map(|n| n as u32),
                config.max_steps as u32,
            ),
            linger: LingerTracker::default(),
            hearts: HeartTracker::default(),
            attached: attach(&config, state, objective),
            steps: 0,
            ended: false,
            config,
        };
        track.begin(state);
        Ok(track)
    }

    /// The next episode starts at `state` (as a training reset: the critic and the end conditions forget the
    /// last one).
    pub fn restart(&mut self, state: &GameView, objective: &Objective) {
        self.attached = attach(&self.config, state, objective);
        self.begin(state);
    }

    fn begin(&mut self, state: &GameView) {
        self.critic.clear();
        self.end_conditions.clear();
        self.hearts.reset(&HeartView::of(state));
        let attached = &self.attached;
        self.linger.reset(&LingerState::of(state, &attached.objective), || {
            linger_wavefront(state, &attached.objective, attached.wavefront.clone())
        });
        self.steps = 0;
        self.ended = false;
    }

    /// The last step ended the episode: [`RewardTrack::restart`] before the next one.
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// Scores the step `change` that ended in `state` (cheated items applied), whose route objective is now
    /// `objective`.
    pub fn step(
        &mut self,
        map: &GameMap,
        change: &Change,
        state: &GameView,
        objective: &Objective,
    ) -> Result<Scored, EnvError> {
        let taken = self.hearts.taken;
        self.hearts.update(&HeartView::of(state), change.frames);
        let curr = attach(&self.config, state, objective);
        let prev = std::mem::replace(&mut self.attached, curr);
        let curr = &self.attached;
        self.linger.update(
            &LingerState::of(&change.previous, &prev.objective),
            &LingerState::of(state, &curr.objective),
            &LingerStep::of(change),
            self.hearts.taken - taken,
            || linger_wavefront(state, &curr.objective, curr.wavefront.clone()),
        );
        self.steps += 1;
        let scored = score_step(
            &self.config,
            &mut self.critic,
            &mut self.end_conditions,
            &Critique {
                change,
                map,
                prev_objective: &prev.objective,
                objective: &curr.objective,
                prev_wavefront: prev.wavefront.as_deref(),
                quiet_steps: self.linger.tail_steps(),
            },
            self.config.reward.shaping_scale,
            self.steps,
            state,
        )?;
        self.ended = scored.ending.is_some();
        Ok(scored)
    }
}
