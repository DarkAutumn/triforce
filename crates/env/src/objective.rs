//! The env's objective attachment (`ZeldaEnv._attach_objectives`): the route objective of a state plus, with
//! PBRS on, the wavefront to its targets.

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

/// The objective of `state` (`prev` is the previous state of the episode) and, with `pbrs`, the wavefront to its
/// PBRS targets (locked doors count as open while Link holds a key in a dungeon).  `nearby_items` adds the tiles
/// overlapping dropped hearts and clocks to the PBRS targets.
pub fn attach(
    gc: &mut GameCompletion,
    prev: Option<&GameView>,
    state: &GameView,
    pbrs: bool,
    nearby_items: bool,
) -> Attached {
    let mut objective = gc.objectives(prev, state);
    if nearby_items {
        for it in &state.items {
            if it.id == item::HEART || it.id == item::CLOCK {
                objective
                    .pbrs_targets
                    .extend(it.link_overlap_tiles().into_iter().map(Target::Tile));
            }
        }
    }
    let wavefront = (pbrs && !state.game_over()).then(|| {
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
