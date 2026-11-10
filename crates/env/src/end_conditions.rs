//! End conditions for specialist episodes and their deny-by-default classification
//! (`triforce/end_conditions.py`).
//!
//! | ending | default class |
//! |---|---|
//! | `exit-correct`, `enter-room:<l>:<r>`, `triforce-collected`, `boss-dead-and-heart` | success if listed |
//! | `exit-wrong`, `max-steps`, `no-room-progress`, `no-damage` | soft failure |
//! | `death`, `wallmastered`, `stuck`, `no-progress` | hard failure |
//! | `retreat` (the route step's `retreat` exit: 0x0a S before the white sword) | neutral |
//!
//! Any ending that is neither a listed success nor in the specialist's/central table is an
//! `UnclassifiedEndingError`.

use crate::rewards::{HARD_FAILURE, NEUTRAL, OUTCOMES, SOFT_FAILURE, SUCCESS};
use indexmap::IndexMap;
use std::collections::HashSet;
use std::fmt;
use zelda::enums::enemy;
use zelda::objectives::{Objective, exit_is_correct};
use zelda::{Change, GameMap, MapLocation};

pub const EXIT_CORRECT: &str = "exit-correct";
pub const EXIT_WRONG: &str = "exit-wrong";
pub const TRIFORCE_COLLECTED: &str = "triforce-collected";
pub const BOSS_DEAD_AND_HEART: &str = "boss-dead-and-heart";
pub const DEATH: &str = "death";
pub const WALLMASTERED: &str = "wallmastered";
pub const STUCK: &str = "stuck";
pub const NO_PROGRESS: &str = "no-progress";
pub const NO_ROOM_PROGRESS: &str = "no-room-progress";
pub const MAX_STEPS: &str = "max-steps";
pub const NO_DAMAGE: &str = "no-damage";
pub const RETREAT: &str = "retreat";
pub const ENTER_ROOM_PREFIX: &str = "enter-room:";

pub const SUCCESS_KINDS: [&str; 3] = [EXIT_CORRECT, TRIFORCE_COLLECTED, BOSS_DEAD_AND_HEART];

/// Central table of the other endings; successes come only from each specialist's `success` list.
pub const DEFAULT_ENDINGS: [(&str, &str); 9] = [
    (EXIT_WRONG, SOFT_FAILURE),
    (MAX_STEPS, SOFT_FAILURE),
    (NO_ROOM_PROGRESS, SOFT_FAILURE),
    (NO_DAMAGE, SOFT_FAILURE),
    (DEATH, HARD_FAILURE),
    (WALLMASTERED, HARD_FAILURE),
    (STUCK, HARD_FAILURE),
    (NO_PROGRESS, HARD_FAILURE),
    (RETREAT, NEUTRAL),
];

/// An episode ended for a reason the specialist doesn't classify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnclassifiedEndingError(pub String);

impl fmt::Display for UnclassifiedEndingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnclassifiedEndingError {}

/// 'enter-room:2:0x0e' (optionally ':cave') -> MapLocation.  The room needs its `0x` (a bare '77' would read as hex
/// here but decimal in a `rooms:` entry, so it's rejected).
pub fn parse_enter_room(kind: &str) -> Result<MapLocation, String> {
    let bad = || format!("Bad success kind {kind:?}; expected enter-room:<level>:0x<room>[:cave]");
    let rest = kind.strip_prefix(ENTER_ROOM_PREFIX).ok_or_else(bad)?;
    let parts: Vec<&str> = rest.split(':').collect();
    if !(2..=3).contains(&parts.len()) || (parts.len() == 3 && parts[2] != "cave") {
        return Err(bad());
    }
    let level: u8 = parts[0].trim().parse().map_err(|_| bad())?;
    let hex = parts[1].trim();
    let hex = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .ok_or_else(bad)?;
    let room = u8::from_str_radix(hex, 16).map_err(|_| bad())?;
    Ok(MapLocation::new(level, room, parts.len() == 3))
}

/// Canonical ending name for entering `location`.
pub fn enter_room_name(location: MapLocation) -> String {
    format!(
        "{ENTER_ROOM_PREFIX}{}:0x{:02x}{}",
        location.level,
        location.value,
        if location.in_cave { ":cave" } else { "" }
    )
}

/// The canonical name of a success kind, or an error for an unknown one.
pub fn validate_success_kind(kind: &str) -> Result<String, String> {
    if kind.starts_with(ENTER_ROOM_PREFIX) {
        return parse_enter_room(kind).map(enter_room_name);
    }
    if !SUCCESS_KINDS.contains(&kind) {
        return Err(format!(
            "Unknown success kind {kind:?}; expected one of {SUCCESS_KINDS:?} or enter-room:<l>:<r>"
        ));
    }
    Ok(kind.to_owned())
}

/// success / hard-failure / soft-failure / neutral for an ending (deny-by-default).
pub fn classify_ending(
    ending: &str,
    success: &[String],
    endings: &IndexMap<String, String>,
) -> Result<&'static str, UnclassifiedEndingError> {
    if success.iter().any(|s| s == ending) {
        return Ok(SUCCESS);
    }
    if let Some(outcome) = endings.get(ending) {
        return OUTCOMES.into_iter().find(|o| o == outcome).ok_or_else(|| {
            UnclassifiedEndingError(format!("Bad outcome {outcome:?} for ending {ending:?}"))
        });
    }
    if let Some((_, outcome)) = DEFAULT_ENDINGS.iter().find(|(e, _)| *e == ending) {
        return Ok(outcome);
    }
    let mut sorted = success.to_vec();
    sorted.sort();
    Err(UnclassifiedEndingError(format!(
        "Ending {ending:?} is not classified (success={sorted:?}, overrides={endings:?}); \
         list it in the specialist's success/endings"
    )))
}

/// What an end condition sees after one step.
pub struct StepView<'a> {
    pub change: &'a Change,
    /// The objective of `change.previous` (decides which exits are correct).
    pub prev_objective: &'a Objective,
    pub map: &'a GameMap,
    /// The shared linger tracker's tail steps, after this step was credited.
    pub linger_tail_steps: u32,
}

impl StepView<'_> {
    fn exit_is_correct(&self, location: MapLocation) -> bool {
        exit_is_correct(
            self.map,
            &self.change.previous,
            self.prev_objective,
            location,
        )
    }
}

/// One end condition (`ZeldaEndCondition` subclasses).
#[derive(Clone, Debug)]
pub enum EndCondition {
    /// Link died.
    GameOver,
    /// A Wallmaster dragged Link out of the room (a jump of more than one room).
    Wallmastered,
    /// Picked up a triforce piece.
    GainedTriforce,
    /// Collected a heart container (it appears only after the boss dies).
    BossDeadAndHeart,
    /// Room transitions: wrong exits, `enter-room` successes, leaving the family (`one_room`: any room).
    RoomExit {
        rooms: HashSet<MapLocation>,
        enter_rooms: HashSet<MapLocation>,
        one_room: bool,
    },
    /// Link stuck in one position, or no new correct room for longer than the specialist's `max_steps` (v1
    /// Timeout; in a one_room episode the `max-steps` truncation always comes first).
    Timeout {
        position_timeout: u32,
        no_progress_timeout: u32,
        position_duration: u32,
        last_progress: u32,
        entered_rooms: HashSet<MapLocation>,
    },
    /// The shared linger tracker's tail reached the limit (a bootstrapping truncation).
    NoRoomProgress { no_progress_steps: u32 },
    /// `limit` actions in one room without damage dealt (an observed hit: on the step it lands) or taken (a
    /// bootstrapping truncation): ends attack loops that never connect.
    NoDamage { limit: u32, quiet: u32 },
}

impl EndCondition {
    pub fn timeout(max_steps: u32) -> EndCondition {
        EndCondition::Timeout {
            position_timeout: 50,
            no_progress_timeout: max_steps,
            position_duration: 0,
            last_progress: 0,
            entered_rooms: HashSet::new(),
        }
    }

    /// Called at reset.
    pub fn clear(&mut self) {
        if let EndCondition::Timeout {
            position_duration,
            last_progress,
            entered_rooms,
            ..
        } = self
        {
            *position_duration = 0;
            *last_progress = 0;
            entered_rooms.clear();
        } else if let EndCondition::NoDamage { quiet, .. } = self {
            *quiet = 0;
        }
    }

    /// The ending name if the episode ended this step.
    pub fn check(&mut self, v: &StepView) -> Option<String> {
        let (prev, curr) = (&v.change.previous, &v.change.state);
        match self {
            EndCondition::GameOver => curr.game_over().then(|| DEATH.to_owned()),
            EndCondition::Wallmastered => (prev.full_location() != curr.full_location()
                && !v.change.gained_triforce()
                && prev.enemies.iter().any(|e| e.id == enemy::WALLMASTER)
                && prev
                    .full_location()
                    .manhattan_distance(curr.full_location())
                    > 1)
            .then(|| WALLMASTERED.to_owned()),
            EndCondition::GainedTriforce => v
                .change
                .gained_triforce()
                .then(|| TRIFORCE_COLLECTED.to_owned()),
            EndCondition::BossDeadAndHeart => {
                (curr.max_health() > prev.max_health()).then(|| BOSS_DEAD_AND_HEART.to_owned())
            }
            EndCondition::RoomExit {
                rooms,
                enter_rooms,
                one_room,
            } => {
                if prev.full_location() == curr.full_location() {
                    return None;
                }
                let location = curr.full_location();
                if !v.exit_is_correct(location) {
                    if v.prev_objective.retreats.contains(&location) {
                        return Some(RETREAT.to_owned());
                    }
                    return Some(EXIT_WRONG.to_owned());
                }
                if enter_rooms.contains(&location) {
                    return Some(enter_room_name(location));
                }
                if rooms.contains(&location) && !*one_room {
                    return None;
                }
                Some(EXIT_CORRECT.to_owned())
            }
            EndCondition::Timeout {
                position_timeout,
                no_progress_timeout,
                position_duration,
                last_progress,
                entered_rooms,
            } => {
                if prev.link.position == curr.link.position && v.change.observed.hits() == 0 {
                    *position_duration += 1;
                    if *position_duration >= *position_timeout {
                        return Some(STUCK.to_owned());
                    }
                } else {
                    *position_duration = 0;
                }
                let location = curr.full_location();
                if prev.full_location() == location {
                    *last_progress += 1;
                } else if v.exit_is_correct(location) && !entered_rooms.contains(&location) {
                    entered_rooms.insert(location);
                    *last_progress = 0;
                }
                (*last_progress > *no_progress_timeout).then(|| NO_PROGRESS.to_owned())
            }
            EndCondition::NoRoomProgress { no_progress_steps } => {
                (v.linger_tail_steps >= *no_progress_steps).then(|| NO_ROOM_PROGRESS.to_owned())
            }
            EndCondition::NoDamage { limit, quiet } => {
                let damage = v.change.health_lost > 0.0 || !v.change.observed.enemies_hit.is_empty();
                if damage || prev.full_location() != curr.full_location() {
                    *quiet = 0;
                } else {
                    *quiet += 1;
                }
                (*quiet >= *limit).then(|| NO_DAMAGE.to_owned())
            }
        }
    }
}

/// The end conditions of a specialist, in priority order (`build_end_conditions`).  Triforce and heart-container
/// conditions only for specialists that list them as success; `no_progress_steps` replaces Timeout with the
/// shared-linger NoRoomProgress (which the env truncates rather than terminates); `no_damage_steps` adds NoDamage
/// last (also a truncation). Timeout's no-new-room limit is `max_steps`.
#[derive(Clone, Debug)]
pub struct EndConditions(pub Vec<EndCondition>);

impl EndConditions {
    pub fn new(
        rooms: &[MapLocation],
        success: &[String],
        one_room: bool,
        no_progress_steps: Option<u32>,
        no_damage_steps: Option<u32>,
        max_steps: u32,
    ) -> EndConditions {
        let enter_rooms = success
            .iter()
            .filter(|k| k.starts_with(ENTER_ROOM_PREFIX))
            .map(|k| parse_enter_room(k).expect("validated success kind"))
            .collect();
        let mut list = vec![EndCondition::GameOver, EndCondition::Wallmastered];
        if success.iter().any(|s| s == TRIFORCE_COLLECTED) {
            list.push(EndCondition::GainedTriforce);
        }
        if success.iter().any(|s| s == BOSS_DEAD_AND_HEART) {
            list.push(EndCondition::BossDeadAndHeart);
        }
        list.push(EndCondition::RoomExit {
            rooms: rooms.iter().copied().collect(),
            enter_rooms,
            one_room,
        });
        list.push(match no_progress_steps {
            None => EndCondition::timeout(max_steps),
            Some(n) => EndCondition::NoRoomProgress {
                no_progress_steps: n,
            },
        });
        if let Some(limit) = no_damage_steps {
            list.push(EndCondition::NoDamage { limit, quiet: 0 });
        }
        EndConditions(list)
    }

    pub fn clear(&mut self) {
        self.0.iter_mut().for_each(EndCondition::clear);
    }

    /// The first ending in priority order (later conditions are not checked once one fires).
    pub fn check(&mut self, view: &StepView) -> Option<String> {
        self.0.iter_mut().find_map(|c| c.check(view))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_kinds_and_enter_room_names() {
        assert_eq!(
            validate_success_kind("exit-correct").unwrap(),
            "exit-correct"
        );
        assert_eq!(
            validate_success_kind("enter-room:2:0x0E").unwrap(),
            "enter-room:2:0x0e"
        );
        assert!(validate_success_kind("enter-room:0:77:cave").is_err());
        assert_eq!(
            validate_success_kind("enter-room:0:0x77:cave").unwrap(),
            "enter-room:0:0x77:cave"
        );
        assert!(validate_success_kind("enter-room:2").is_err());
        assert!(validate_success_kind("exit-sideways").is_err());
        assert_eq!(
            parse_enter_room("enter-room:2:0x0e").unwrap(),
            MapLocation::new(2, 0x0e, false)
        );
    }

    #[test]
    fn deny_by_default_classification() {
        let success = vec!["exit-correct".to_owned()];
        let mut endings = IndexMap::new();
        assert_eq!(
            classify_ending("exit-correct", &success, &endings),
            Ok(SUCCESS)
        );
        assert_eq!(
            classify_ending("death", &success, &endings),
            Ok(HARD_FAILURE)
        );
        assert_eq!(
            classify_ending("max-steps", &success, &endings),
            Ok(SOFT_FAILURE)
        );
        assert!(classify_ending("triforce-collected", &success, &endings).is_err());
        endings.insert("exit-wrong".to_owned(), HARD_FAILURE.to_owned());
        assert_eq!(
            classify_ending("exit-wrong", &success, &endings),
            Ok(HARD_FAILURE)
        );
    }

    #[test]
    fn condition_order() {
        let success = vec![
            "triforce-collected".to_owned(),
            "boss-dead-and-heart".to_owned(),
        ];
        let c = EndConditions::new(
            &[MapLocation::new(1, 0x35, false)],
            &success,
            false,
            Some(5),
            Some(7),
            2000,
        );
        let kinds: Vec<&str> =
            c.0.iter()
                .map(|c| match c {
                    EndCondition::GameOver => "game-over",
                    EndCondition::Wallmastered => "wallmastered",
                    EndCondition::GainedTriforce => "triforce",
                    EndCondition::BossDeadAndHeart => "heart",
                    EndCondition::RoomExit { .. } => "exit",
                    EndCondition::Timeout { .. } => "timeout",
                    EndCondition::NoRoomProgress { .. } => "no-room-progress",
                    EndCondition::NoDamage { .. } => "no-damage",
                })
                .collect();
        assert_eq!(
            kinds,
            [
                "game-over",
                "wallmastered",
                "triforce",
                "heart",
                "exit",
                "no-room-progress",
                "no-damage"
            ]
        );
    }
}
