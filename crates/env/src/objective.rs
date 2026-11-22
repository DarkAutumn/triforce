//! The env's objective attachment (`ZeldaEnv._attach_objectives`): the route objective of a state plus, with
//! PBRS (or the `target` observation) on, the wavefront to its targets.

use crate::config::{RewardConfig, SpecialistConfig};
use std::sync::Arc;
use zelda::enums::{Direction, Tile, enemy, item};
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

/// A clock Link must not take (round 26, 7:0x0d): one lying in a room while a Wallmaster lives.  Under the clock a
/// Wallmaster that comes out stays where it starts, two squares inside the wall, out of the sword's and the
/// boomerang's reach, and the room can't be cleared any more.
fn clock_is_a_trap(state: &GameView, id: i16) -> bool {
    id == item::CLOCK && state.enemies.iter().any(|e| e.id == enemy::WALLMASTER)
}

/// The tiles on which Link touches such a clock ([`clock_is_a_trap`]): the path search goes round them and no
/// target lies on them (a hand dies on the wall line, so its clock lies on a spot that would bring the next out).
fn trap_tiles(state: &GameView) -> Vec<Tile> {
    state
        .items
        .iter()
        .filter(|it| clock_is_a_trap(state, it.id))
        .flat_map(|it| it.link_overlap_tiles())
        .collect()
}

/// Whether the dropped item `id` is a PBRS target under `reward` with Link at `state`'s health:
/// `nearby_items` takes hearts and clocks; `item_targets` every dropped item, hearts and fairies only below max
/// health (and with `drops_first`, the family's or the route step's: [`drops_first`], bombs only below the bomb
/// maximum); `heal_targets` hearts and fairies alone, below max health. Under a route's bomb floor (`bag`,
/// `Objective::bag_drops`, round 37) bombs are one below the bomb maximum and only there, in every family. The
/// room's own treasure is never one (it is the objective's business), nor a clock while a Wallmaster lives
/// ([`clock_is_a_trap`]).
fn is_item_target(reward: &RewardConfig, drops_first: bool, bag: bool, state: &GameView, id: i16) -> bool {
    if id < 0 || clock_is_a_trap(state, id) {
        return false;
    }
    if bag && id == item::BOMBS {
        return state.bombs() < state.bomb_max();
    }
    let heals = id == item::HEART || id == item::FAIRY;
    if reward.item_targets {
        if drops_first && id == item::BOMBS {
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

/// Whether the wanted drops come first for `objective`: the family's `reward.drops_first` (round 25) or the route
/// step's `drops: first` (round 31, 6:0x78: `Objective::drops_first`).
fn drops_first(reward: &RewardConfig, objective: &Objective) -> bool {
    reward.drops_first || objective.drops_first
}

/// The tiles of the dropped items `reward` (and, `bag`, the route's bomb floor) makes PBRS targets.
fn item_targets<'a>(
    reward: &'a RewardConfig,
    drops_first: bool,
    bag: bool,
    state: &'a GameView,
) -> impl Iterator<Item = Target> + 'a {
    state
        .items
        .iter()
        .filter(move |it| is_item_target(reward, drops_first, bag, state, it.id))
        .flat_map(|it| it.link_overlap_tiles().into_iter().map(Target::Tile))
}

/// Once `objective` is no fight (the room's monsters are dead, or a room only crossed), the drops that come first
/// as the only targets until taken or gone: with [`drops_first`] the tiles of the wanted drops lying in the room
/// ([`wanted_drop`]; round 25, the owner: "make it a primary objective to pick them up on win"); under the route's
/// bomb floor (`Objective::bag_drops`, round 37) the dropped bombs below the bomb maximum, while Link can reach them
/// (the owner, Nov 22: no push out of the room past bombs he could take). None without any.
fn drops_first_targets(reward: &RewardConfig, state: &GameView, objective: &Objective) -> Option<Vec<Target>> {
    let first = drops_first(reward, objective);
    if !(first || objective.bag_drops) || objective.kind == ObjectiveKind::Fight {
        return None;
    }
    let tiles: Vec<Target> = state
        .items
        .iter()
        .filter(|it| wanted_drop(state, it.id) && (first || it.id == item::BOMBS))
        .flat_map(|it| it.link_overlap_tiles().into_iter().map(Target::Tile))
        .collect();
    if tiles.is_empty() || (!first && wavefront_to(state, objective, &tiles).get(state.link.tile()).is_none()) {
        return None;
    }
    Some(tiles)
}

/// The wavefront to `objective`'s `targets` (locked doors count as open while Link holds a key in a dungeon; the
/// grid-point skips are jumps when the objective's specialist has clips on; a clock that must not be taken is
/// walked round, [`trap_tiles`]).
fn wavefront_to(state: &GameView, objective: &Objective, targets: &[Target]) -> Wavefront {
    let locked: Vec<Direction> = if state.keys() > 0 && state.level() > 0 {
        [Direction::N, Direction::S, Direction::E, Direction::W]
            .into_iter()
            .filter(|&d| state.is_door_locked(d))
            .collect()
    } else {
        Vec::new()
    };
    state.room().wavefront(targets, &trap_tiles(state), &locked, objective.clips)
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
    let first = drops_first(reward, &objective);
    match drops_first_targets(reward, state, &objective) {
        Some(drops) => objective.pbrs_targets = drops.into_iter().collect(),
        None => objective.pbrs_targets.extend(item_targets(reward, first, objective.bag_drops, state)),
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
        None => {
            let first = drops_first(reward, objective);
            objective.pbrs_targets.iter().copied().chain(item_targets(reward, first, objective.bag_drops, state)).collect()
        }
    };
    targets.sort_unstable();
    targets.dedup();
    Some(wavefront_to(state, objective, &targets))
}

/// Whether a dropped item pulls Link away from `objective` (the aimless score's `pulled`, `zelda::aimless`): one the
/// shaping of `reward` makes a target, or one he wants ([`wanted_drop`]: a heart or fairy below full health, bombs
/// below his maximum). Dropped items vanish; walking to one is no aimless walk (the owner, round 38: "walking towards
/// a heart that disappears shouldn't count").
pub fn pulled(reward: &RewardConfig, objective: &Objective, state: &GameView) -> bool {
    let first = drops_first(reward, objective);
    state
        .items
        .iter()
        .any(|it| wanted_drop(state, it.id) || is_item_target(reward, first, objective.bag_drops, state, it.id))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use zelda::aimless::AimlessState;
    use zelda::GameMap;
    use zelda::tables::RAM_SIZE;

    const LEVEL: usize = 0x10;
    const ROOM_ID: usize = 0xEB;
    const GAME_MODE: usize = 0x12;
    const OBJ_X: usize = 0x70;
    const OBJ_Y: usize = 0x84;
    const OBJ_STATE: usize = 0xAC;
    const OBJ_TYPE: usize = 0x34F;

    /// `item_targets` makes every dropped item a target, a clock too; with a Wallmaster alive the clock is left
    /// out (taken, it freezes the hands in their wall for the rest of the visit) and the rupee stays.
    #[test]
    fn a_clock_is_no_target_while_a_wallmaster_lives() {
        let map = Arc::new(GameMap::builtin());
        let mut ram = [0u8; RAM_SIZE];
        (ram[LEVEL], ram[ROOM_ID], ram[GAME_MODE]) = (7, 0x0d, 5);
        (ram[OBJ_X], ram[OBJ_Y]) = (0x30, 0x9D);
        // Dropped items are object type $60 with the item in ObjState: a clock where a hand died, a rupee.
        for (slot, kind, x, y) in [(3, item::CLOCK, 0x20, 0xA6), (5, item::RUPEE, 0x60, 0x8D)] {
            (ram[OBJ_TYPE + slot], ram[OBJ_STATE + slot], ram[OBJ_X + slot], ram[OBJ_Y + slot]) = (0x60, kind as u8, x, y);
        }
        let targets = |ram: &[u8; RAM_SIZE], reward: &RewardConfig| -> Vec<Target> {
            item_targets(reward, reward.drops_first, false, &GameView::new(ram, 0, false, map.clone())).collect()
        };
        let all = RewardConfig { item_targets: true, ..Default::default() };
        let near = RewardConfig { nearby_items: true, ..Default::default() };
        let (clock_and_rupee, clock) = (targets(&ram, &all), targets(&ram, &near));
        assert_eq!((clock_and_rupee.len(), clock.len()), (8, 4));
        // A Wallmaster waiting in its wall (state 0).
        (ram[OBJ_TYPE + 4], ram[OBJ_STATE + 4]) = (enemy::WALLMASTER, 0);
        let rupee = targets(&ram, &all);
        assert_eq!(rupee.len(), 4);
        assert!(rupee.iter().all(|t| clock_and_rupee.contains(t) && !clock.contains(t)));
        assert!(targets(&ram, &near).is_empty());
    }

    /// Round 38, the aimless score: a dropped heart pulls a hurt Link whatever his family's shaping (it vanishes;
    /// walking to it is no aimless walk), so no decision is calm while it lies there; at full health it pulls nobody
    /// and the empty screen is calm again; a rupee pulls only a family whose shaping makes it a target.
    #[test]
    fn a_heart_that_can_vanish_is_no_calm_goal() {
        const HEARTS: usize = 0x66F;
        const PARTIAL: usize = 0x670;
        let map = Arc::new(GameMap::builtin());
        let mut ram = [0u8; RAM_SIZE];
        (ram[LEVEL], ram[ROOM_ID], ram[GAME_MODE]) = (0, 0x77, 5);
        (ram[OBJ_X], ram[OBJ_Y]) = (0x40, 0x8D);
        (ram[OBJ_TYPE + 3], ram[OBJ_STATE + 3], ram[OBJ_X + 3], ram[OBJ_Y + 3]) = (0x60, item::HEART as u8, 0x60, 0x8D);
        (ram[HEARTS], ram[PARTIAL]) = (0x21, 0x00);
        let objective = Objective {
            kind: ObjectiveKind::Move,
            targets: BTreeSet::new(),
            next_rooms: [zelda::MapLocation::new(0, 0x76, false)].into_iter().collect(),
            pbrs_targets: [Target::Tile((-1, 10))].into_iter().collect(),
            retreats: BTreeSet::new(),
            clips: false,
            treasure_refused: false,
            drops_first: false,
            bag_drops: false,
        };
        let plain = RewardConfig::default();
        let view = |ram: &[u8; RAM_SIZE]| GameView::new(ram, 0, false, map.clone());
        let hurt = view(&ram);
        assert!(!hurt.is_health_full());
        assert!(pulled(&plain, &objective, &hurt));
        assert!(!AimlessState::of(&hurt, &objective, true).quiet);

        ram[PARTIAL] = 0xFF;
        ram[HEARTS] = 0x22;
        let full = view(&ram);
        assert!(full.is_health_full());
        assert!(!pulled(&plain, &objective, &full));
        assert!(AimlessState::of(&full, &objective, false).quiet, "an empty screen with an exit to walk to");

        ram[OBJ_STATE + 3] = item::RUPEE as u8;
        let rupee = view(&ram);
        assert!(!pulled(&plain, &objective, &rupee));
        assert!(pulled(&RewardConfig { item_targets: true, ..Default::default() }, &objective, &rupee));
    }
}
