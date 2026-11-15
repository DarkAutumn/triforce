//! The env's objective attachment (`ZeldaEnv._attach_objectives`): the route objective of a state plus, with
//! PBRS (or the `target` observation) on, the wavefront to its targets.

use crate::config::{RewardConfig, SpecialistConfig};
use std::sync::Arc;
use zelda::enums::{Direction, item};
use zelda::game::GameView;
use zelda::objectives::{GameCompletion, Objective, ObjectiveKind};
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
/// health (and with `drops_first`, bombs only below the bomb maximum); `heal_targets` hearts and fairies alone,
/// below max health. The room's own treasure is never one (it is
/// the objective's business).
fn is_item_target(reward: &RewardConfig, state: &GameView, id: i16) -> bool {
    if id < 0 {
        return false;
    }
    let heals = id == item::HEART || id == item::FAIRY;
    if reward.item_targets {
        if reward.drops_first && id == item::BOMBS {
            return state.bombs() < state.bomb_max();
        }
        return !heals || !state.is_health_full();
    }
    if reward.heal_targets && heals && !state.is_health_full() {
        return true;
    }
    reward.nearby_items && (id == item::HEART || id == item::CLOCK)
}

/// Whether the dropped item `id` is worth a detour to Link at `state` (round 25): bombs below his bomb maximum, a
/// heart or fairy below max health.
pub fn wanted_drop(state: &GameView, id: i16) -> bool {
    match id {
        item::BOMBS => state.bombs() < state.bomb_max(),
        item::HEART | item::FAIRY => !state.is_health_full(),
        _ => false,
    }
}

/// Whether `reward.drops_first` puts the dropped item `id` before the room's treasure and exits ([`wanted_drop`]).
fn is_wanted_drop(reward: &RewardConfig, state: &GameView, id: i16) -> bool {
    reward.drops_first && wanted_drop(state, id)
}

/// The tiles of the dropped items `reward` makes PBRS targets.
fn item_targets<'a>(reward: &'a RewardConfig, state: &'a GameView) -> impl Iterator<Item = Target> + 'a {
    state
        .items
        .iter()
        .filter(|it| is_item_target(reward, state, it.id))
        .flat_map(|it| it.link_overlap_tiles().into_iter().map(Target::Tile))
}

/// With `reward.drops_first`, once `objective` is no fight (the room's monsters are dead), the tiles of the wanted
/// drops lying in the room ([`is_wanted_drop`]): they are the only targets until taken or gone (round 25, the
/// owner: "make it a primary objective to pick them up on win"). None without any.
fn drops_first_targets(reward: &RewardConfig, state: &GameView, objective: &Objective) -> Option<Vec<Target>> {
    if !reward.drops_first || objective.kind == ObjectiveKind::Fight {
        return None;
    }
    let tiles: Vec<Target> = state
        .items
        .iter()
        .filter(|it| is_wanted_drop(reward, state, it.id))
        .flat_map(|it| it.link_overlap_tiles().into_iter().map(Target::Tile))
        .collect();
    (!tiles.is_empty()).then_some(tiles)
}

/// The wavefront to `objective`'s `targets` (locked doors count as open while Link holds a key in a dungeon; the
/// grid-point skips are jumps when the objective's specialist has clips on).
fn wavefront_to(state: &GameView, objective: &Objective, targets: &[Target]) -> Wavefront {
    let locked: Vec<Direction> = if state.keys() > 0 && state.level() > 0 {
        [Direction::N, Direction::S, Direction::E, Direction::W]
            .into_iter()
            .filter(|&d| state.is_door_locked(d))
            .collect()
    } else {
        Vec::new()
    };
    state.room().wavefront(targets, &[], &locked, objective.clips)
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
/// targets join its targets (or, with `drops_first` after the fight, replace them) and, with `wavefront`, the
/// wavefront to them is built.
pub fn attach_objective(
    mut objective: Objective,
    state: &GameView,
    reward: &RewardConfig,
    wavefront: bool,
) -> Attached {
    match drops_first_targets(reward, state, &objective) {
        Some(drops) => objective.pbrs_targets = drops.into_iter().collect(),
        None => objective.pbrs_targets.extend(item_targets(reward, state)),
    }
    let wavefront = (wavefront && !state.game_over()).then(|| {
        let targets: Vec<Target> = objective.pbrs_targets.iter().copied().collect();
        Arc::new(wavefront_to(state, &objective, &targets))
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
    let mut targets: Vec<Target> = match drops_first_targets(reward, state, objective) {
        Some(drops) => drops,
        None => objective.pbrs_targets.iter().copied().chain(item_targets(reward, state)).collect(),
    };
    targets.sort_unstable();
    targets.dedup();
    Some(wavefront_to(state, objective, &targets))
}

/// Direction bits (E 1, W 2, S 4, N 8) of the sides `objective`'s next rooms lie on from `state`'s room (a maze
/// room's next step): the `next_room_*` observation.
pub fn exit_bits(state: &GameView, objective: &Objective) -> u8 {
    let location = state.full_location();
    let maze_step = state.peek(zelda::game_map::MAZE_STEP);
    objective
        .next_rooms
        .iter()
        .fold(0, |bits, next| bits | state.game_map().exit_direction(location, *next, maze_step).value())
}
