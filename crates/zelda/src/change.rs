//! What changed over one option, with look-ahead credit for delayed weapons (`triforce/state_change.py`).

use crate::actions::{Action, ActionMask};
use crate::blocks::ProjectileBlock;
use crate::enums::{AnimationKind, AnimationState};
use crate::executor::{Interrupt, Machine, OptionResult};
use crate::game::GameView;
use crate::ledger::{FutureCreditLedger, Prediction, add_hit};
use crate::nes::Nes;
use crate::vars::Var;
use std::sync::Arc;

/// Safety cap on one look-ahead.
pub const LOOK_AHEAD_MAX_FRAMES: u32 = 400;

/// Weapons whose activation runs a look-ahead, in order.
pub const LOOK_AHEAD_WEAPONS: [AnimationKind; 8] = [
    AnimationKind::Beams,
    AnimationKind::Magic,
    AnimationKind::Bomb1,
    AnimationKind::Bomb2,
    AnimationKind::Flame1,
    AnimationKind::Flame2,
    AnimationKind::Arrow,
    AnimationKind::Boomerang,
];

/// The changes between two consecutive controllable states.
#[derive(Clone, Debug)]
pub struct Change {
    pub action: Action,
    pub frames: u32,
    pub interrupt: Option<Interrupt>,
    pub control_frames: u32,
    /// Confirmed shield blocks during the option (never changed by look-ahead).
    pub projectile_blocks: Vec<ProjectileBlock>,
    pub previous: Arc<GameView>,
    pub state: Arc<GameView>,
    /// The mask of `state` (as `GameSession.act` sets it).
    pub action_mask: ActionMask,
    pub health_lost: f64,
    pub health_gained: f64,
    /// (slot, damage), observed plus predicted, minus earlier predictions.
    pub enemies_hit: Vec<(u8, i32)>,
    pub enemies_stunned: Vec<u8>,
    pub items_gained: Vec<i16>,
}

impl Change {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<N: Nes>(
        m: &mut Machine<N>,
        prev: Arc<GameView>,
        curr: Arc<GameView>,
        action: Action,
        option: OptionResult,
        action_mask: ActionMask,
        ledger: &mut FutureCreditLedger,
        look_ahead: bool,
    ) -> Change {
        let same_max = prev.max_health() == curr.max_health();
        let health_lost = if same_max {
            (prev.health() - curr.health()).max(0.0)
        } else {
            ((prev.max_health() - curr.max_health()) as f64).max(0.0)
        };
        let health_gained = if same_max {
            (curr.health() - prev.health()).max(0.0)
        } else {
            ((curr.max_health() - prev.max_health()) as f64).max(0.0)
        };
        let mut change = Change {
            action,
            frames: option.frames,
            interrupt: option.interrupt,
            control_frames: option.control_frames,
            projectile_blocks: option.projectile_blocks,
            previous: prev,
            state: curr,
            action_mask,
            health_lost,
            health_gained,
            enemies_hit: Vec::new(),
            enemies_stunned: Vec::new(),
            items_gained: Vec::new(),
        };
        if change.changed_location() {
            ledger.clear();
        } else {
            let (prev, curr) = (change.previous.clone(), change.state.clone());
            observe_damage(
                &prev,
                &curr,
                &mut change.enemies_hit,
                &mut change.enemies_stunned,
            );
            observe_items(&prev, &curr, &mut change.items_gained);
            ledger.discount(
                curr.frame,
                &mut change.enemies_hit,
                &mut change.enemies_stunned,
                &mut change.items_gained,
            );
            if look_ahead {
                for weapon in LOOK_AHEAD_WEAPONS {
                    if curr.animation_state(weapon) == AnimationState::Active
                        && prev.animation_state(weapon) != AnimationState::Active
                    {
                        change.predict(m, &curr, weapon, ledger);
                    }
                }
            }
        }
        change
    }

    pub fn changed_location(&self) -> bool {
        self.previous.full_location() != self.state.full_location()
    }

    pub fn gained_triforce(&self) -> bool {
        self.previous.triforce_pieces() < self.state.triforce_pieces()
            || (!self.previous.triforce_of_power() && self.state.triforce_of_power())
    }

    pub fn blocks(&self) -> usize {
        self.projectile_blocks.len()
    }

    pub fn damage_dealt(&self) -> i32 {
        self.enemies_hit.iter().map(|(_, d)| d).sum()
    }

    pub fn hits(&self) -> usize {
        self.enemies_hit.len()
    }

    /// Simulates until the weapon deactivates on a cloned machine state (no input, invincible hearts, other
    /// weapon slots zeroed, no render, no frame sink, no shield watch), credits and records the prediction, then
    /// restores the machine.
    fn predict<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        start: &Arc<GameView>,
        weapon: AnimationKind,
        ledger: &mut FutureCreditLedger,
    ) {
        let saved = m.nes.snapshot();
        for var in disabled_slots(weapon) {
            m.nes.poke(var.addr(), 0);
        }
        let map = start.game_map().clone();
        let mut curr = start.clone();
        let mut frame = start.frame;
        for _ in 0..LOOK_AHEAD_MAX_FRAMES {
            if curr.game_over()
                || start.location() != curr.location()
                || curr.animation_state(weapon) == AnimationState::Inactive
            {
                break;
            }
            m.nes.poke(Var::HeartsAndContainers.addr(), 0xFF);
            m.nes.run_frame(0, false);
            frame += 1;
            curr = Arc::new(GameView::new(
                &m.ram_image(),
                frame,
                start.just_exited_cave,
                map.clone(),
            ));
        }
        let mut prediction = Prediction {
            frame: start.frame,
            ..Default::default()
        };
        observe_damage(
            start,
            &curr,
            &mut prediction.enemy_hits,
            &mut prediction.enemy_stuns,
        );
        if matches!(weapon, AnimationKind::Boomerang | AnimationKind::Arrow) {
            observe_items(start, &curr, &mut prediction.items);
        }
        for &(slot, dmg) in &prediction.enemy_hits {
            add_hit(&mut self.enemies_hit, slot, dmg);
        }
        self.enemies_stunned
            .extend_from_slice(&prediction.enemy_stuns);
        self.items_gained.extend_from_slice(&prediction.items);
        ledger.add_prediction(prediction);
        m.nes.restore(&saved);
    }
}

/// The look-ahead zeroes every weapon slot but the one it follows.
fn disabled_slots(weapon: AnimationKind) -> Vec<Var> {
    let keep = match weapon {
        AnimationKind::Beams | AnimationKind::Magic => Var::BeamAnimation,
        AnimationKind::Bomb1 | AnimationKind::Flame1 => Var::BombOrFlameAnimation,
        AnimationKind::Bomb2 | AnimationKind::Flame2 => Var::BombOrFlameAnimation2,
        AnimationKind::Arrow => Var::ArrowMagicAnimation,
        AnimationKind::Boomerang => Var::BaitOrBoomerangAnimation,
        AnimationKind::Bait => panic!("no look-ahead for {weapon:?}"),
    };
    [
        Var::BeamAnimation,
        Var::BombOrFlameAnimation,
        Var::BombOrFlameAnimation2,
        Var::BaitOrBoomerangAnimation,
        Var::ArrowMagicAnimation,
    ]
    .into_iter()
    .filter(|&v| v != keep)
    .collect()
}

/// Enemy health drops, kills and new stuns from `prev` to `curr`.
fn observe_damage(
    prev: &GameView,
    curr: &GameView,
    hits: &mut Vec<(u8, i32)>,
    stuns: &mut Vec<u8>,
) {
    for enemy in &prev.enemies {
        if enemy.is_dying() {
            continue;
        }
        match curr.enemy_by_index(enemy.index) {
            Some(now) => {
                let dmg = enemy.health - now.health;
                if dmg > 0 {
                    add_hit(hits, enemy.index, dmg);
                } else if now.is_dying() {
                    add_hit(
                        hits,
                        enemy.index,
                        if enemy.health > 0 { enemy.health } else { 1 },
                    );
                }
                if now.is_stunned() && !enemy.is_stunned() {
                    stuns.push(enemy.index);
                }
            }
            None => add_hit(
                hits,
                enemy.index,
                if enemy.health > 0 { enemy.health } else { 1 },
            ),
        }
    }
}

/// Items gone from `prev` to `curr` before their timers ran out (picked up).
fn observe_items(prev: &GameView, curr: &GameView, items: &mut Vec<i16>) {
    let elapsed = curr.frame.saturating_sub(prev.frame);
    for item in &prev.items {
        if curr.item_by_index(item.index).is_none() && elapsed < item.timer as u64 {
            items.push(item.id);
        }
    }
}
