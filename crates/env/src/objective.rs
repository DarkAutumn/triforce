//! The env's objective attachment (`ZeldaEnv._attach_objectives`): the route objective of a state plus, with
//! PBRS (or the `target` observation) on, the wavefront to its targets.

use crate::config::{RewardConfig, SpecialistConfig};
use std::sync::Arc;
use zelda::enums::{Direction, item};
use zelda::game::GameView;
use zelda::objectives::{GameCompletion, Objective};
use zelda::room::{Target, Wavefront};

/// A state's objective and PBRS wavefront.
#[derive(Clone, Debug)]
pub struct Attached {
    pub objective: Objective,
    /// None with PBRS and `obs.target` off, or on game over.
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

/// The tiles of the dropped items `reward` makes PBRS targets.
fn item_targets<'a>(reward: &'a RewardConfig, state: &'a GameView) -> impl Iterator<Item = Target> + 'a {
    state
        .items
        .iter()
        .filter(|it| is_item_target(reward, state, it.id))
        .flat_map(|it| it.link_overlap_tiles().into_iter().map(Target::Tile))
}

/// The wavefront to `targets` (locked doors count as open while Link holds a key in a dungeon).
fn wavefront_to(state: &GameView, targets: &[Target]) -> Wavefront {
    let locked: Vec<Direction> = if state.keys() > 0 && state.level() > 0 {
        [Direction::N, Direction::S, Direction::E, Direction::W]
            .into_iter()
            .filter(|&d| state.is_door_locked(d))
            .collect()
    } else {
        Vec::new()
    };
    state.room().wavefront(targets, &[], &locked)
}

/// Whether a specialist needs the wavefront: for PBRS or for its `target` observation.
pub fn needs_wavefront(config: &SpecialistConfig) -> bool {
    config.reward.pbrs || config.obs.target
}

/// The objective of `state` (`prev` is the previous state of the episode) and, with `wavefront` (see
/// [`needs_wavefront`]), the wavefront to its PBRS targets, dropped items included as `reward.nearby_items` /
/// `reward.item_targets` say.
pub fn attach(
    gc: &mut GameCompletion,
    prev: Option<&GameView>,
    state: &GameView,
    reward: &RewardConfig,
    wavefront: bool,
) -> Attached {
    attach_objective(gc.objectives(prev, state), state, reward, wavefront)
}

/// [`attach`] for a route objective computed elsewhere (a chain's): the dropped items `reward` makes PBRS
/// targets join its targets and, with `wavefront`, the wavefront to them is built.
pub fn attach_objective(
    mut objective: Objective,
    state: &GameView,
    reward: &RewardConfig,
    wavefront: bool,
) -> Attached {
    objective.pbrs_targets.extend(item_targets(reward, state));
    let wavefront = (wavefront && !state.game_over()).then(|| {
        let targets: Vec<Target> = objective.pbrs_targets.iter().copied().collect();
        Arc::new(wavefront_to(state, &targets))
    });
    Attached {
        objective,
        wavefront,
    }
}

/// The wavefront [`attach`] gives a specialist with `reward` for `objective` (the route objective without
/// dropped items), for observations built outside its env (chains).  None on game over.
pub fn target_wavefront(
    objective: &Objective,
    state: &GameView,
    reward: &RewardConfig,
) -> Option<Wavefront> {
    if state.game_over() {
        return None;
    }
    let mut targets: Vec<Target> = objective.pbrs_targets.iter().copied().collect();
    targets.extend(item_targets(reward, state));
    targets.sort_unstable();
    targets.dedup();
    Some(wavefront_to(state, &targets))
}

/// Direction bits (E 1, W 2, S 4, N 8) of the sides `objective`'s next rooms lie on from `state`'s room (a maze
/// room's next step): the `next_room_*` observation and the correct exits of the `edge` clip trigger.
pub fn exit_bits(state: &GameView, objective: &Objective) -> u8 {
    let location = state.full_location();
    let maze_step = state.peek(zelda::game_map::MAZE_STEP);
    objective
        .next_rooms
        .iter()
        .fold(0, |bits, next| bits | state.game_map().exit_direction(location, *next, maze_step).value())
}

/// The session's clip settings for `config` in `state` with `objective` (the trigger, the correct exits).
pub fn clip_settings(config: &SpecialistConfig, state: &GameView, objective: &Objective) -> zelda::clips::ClipSettings {
    zelda::clips::ClipSettings::new(config.clips.action.trigger(), exit_bits(state, objective))
}
