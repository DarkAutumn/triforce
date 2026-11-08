//! The env's objective attachment (`ZeldaEnv._attach_objectives`): the route objective of a state plus, with
//! PBRS on, the wavefront to its targets.

use crate::config::RewardConfig;
use std::sync::Arc;
use zelda::enums::{Direction, item};
use zelda::game::GameView;
use zelda::objectives::{GameCompletion, Objective};
use zelda::room::{Target, Wavefront};

/// A state's objective and PBRS wavefront.
#[derive(Clone, Debug)]
pub struct Attached {
    pub objective: Objective,
    /// None with PBRS off or on game over.
    pub wavefront: Option<Arc<Wavefront>>,
}

/// Whether the dropped item `id` is a PBRS target under `reward` with Link at `state`'s health:
/// `nearby_items` takes hearts and clocks; `item_targets` every dropped item, hearts and fairies only below max
/// health. The room's own treasure is never one (it is the objective's business).
fn is_item_target(reward: &RewardConfig, state: &GameView, id: i16) -> bool {
    if id < 0 {
        return false;
    }
    if reward.item_targets {
        let heals = id == item::HEART || id == item::FAIRY;
        return !heals || !state.is_health_full();
    }
    reward.nearby_items && (id == item::HEART || id == item::CLOCK)
}

/// The objective of `state` (`prev` is the previous state of the episode) and, with `reward.pbrs`, the wavefront
/// to its PBRS targets (locked doors count as open while Link holds a key in a dungeon), dropped items included
/// as `reward.nearby_items` / `reward.item_targets` say.
pub fn attach(
    gc: &mut GameCompletion,
    prev: Option<&GameView>,
    state: &GameView,
    reward: &RewardConfig,
) -> Attached {
    let mut objective = gc.objectives(prev, state);
    for it in &state.items {
        if is_item_target(reward, state, it.id) {
            objective
                .pbrs_targets
                .extend(it.link_overlap_tiles().into_iter().map(Target::Tile));
        }
    }
    let wavefront = (reward.pbrs && !state.game_over()).then(|| {
        let locked: Vec<Direction> = if state.keys() > 0 && state.level() > 0 {
            [Direction::N, Direction::S, Direction::E, Direction::W]
                .into_iter()
                .filter(|&d| state.is_door_locked(d))
                .collect()
        } else {
            Vec::new()
        };
        let targets: Vec<Target> = objective.pbrs_targets.iter().copied().collect();
        Arc::new(state.room().wavefront(&targets, &[], &locked))
    });
    Attached {
        objective,
        wavefront,
    }
}
