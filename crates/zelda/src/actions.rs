//! The v2 action schema and masking (`triforce/actions.py`): kind x direction x duration timed options.

use crate::enums::{ActionKind, Direction};
use crate::game::GameView;
use std::fmt;

pub const ACTION_SCHEMA_VERSION: u32 = 1;

pub const DIRECTIONS: [Direction; 8] = [
    Direction::N,
    Direction::S,
    Direction::W,
    Direction::E,
    Direction::NW,
    Direction::NE,
    Direction::SW,
    Direction::SE,
];
pub const CARDINALS: [Direction; 4] = [Direction::N, Direction::S, Direction::W, Direction::E];
pub const DURATIONS: [u8; 4] = [4, 8, 16, 32];

pub const NUM_KINDS: usize = 7;
pub const NUM_DIRECTIONS: usize = 8;
pub const NUM_DURATIONS: usize = 4;
/// Flat mask layout: `[kind (K) | direction (K x D) | duration (K x U)]`.
pub const MASK_SIZE: usize = NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS + NUM_KINDS * NUM_DURATIONS;

pub fn direction_index(d: Direction) -> Option<usize> {
    DIRECTIONS.iter().position(|&x| x == d)
}

pub fn duration_index(d: u8) -> Option<usize> {
    DURATIONS.iter().position(|&x| x == d)
}

/// MOVE and WAIT have durations.
pub const fn is_timed(kind: ActionKind) -> bool {
    matches!(kind, ActionKind::Move | ActionKind::Wait)
}

/// Every kind except WAIT has a direction.
pub const fn is_directional(kind: ActionKind) -> bool {
    !matches!(kind, ActionKind::Wait)
}

/// Only the boomerang flies diagonally.
pub const fn is_diagonal(kind: ActionKind) -> bool {
    matches!(kind, ActionKind::Boomerang)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionError(pub String);

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ActionError {}

/// A timed option.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Action {
    pub kind: ActionKind,
    pub direction: Direction,
    pub duration: u8,
}

impl Action {
    /// Validated constructor (directional kinds need a direction, cardinal unless diagonal-capable; timed kinds a
    /// duration of 4, 8, 16 or 32).
    pub fn new(
        kind: ActionKind,
        direction: Direction,
        duration: u8,
    ) -> Result<Action, ActionError> {
        if is_directional(kind) {
            if direction_index(direction).is_none() {
                return Err(ActionError(format!("{} requires a direction", kind.name())));
            }
            if !is_diagonal(kind) && !direction.is_cardinal() {
                return Err(ActionError(format!(
                    "{} does not support diagonal {}",
                    kind.name(),
                    direction.name()
                )));
            }
        }
        if is_timed(kind) && duration_index(duration).is_none() {
            return Err(ActionError(format!(
                "{} requires a duration in {DURATIONS:?}",
                kind.name()
            )));
        }
        Ok(Action {
            kind,
            direction,
            duration,
        })
    }

    /// From head outputs, ignoring heads that don't apply.  Errors for out-of-range indices and for
    /// diagonal directions on kinds that can't use them.
    pub fn from_indices(
        kind: usize,
        direction: usize,
        duration: usize,
    ) -> Result<Action, ActionError> {
        let kind = ActionKind::from_index(kind)
            .ok_or_else(|| ActionError(format!("no action kind {kind}")))?;
        let d = if is_directional(kind) {
            *DIRECTIONS
                .get(direction)
                .ok_or_else(|| ActionError(format!("no direction {direction}")))?
        } else {
            Direction::None
        };
        let u = if is_timed(kind) {
            *DURATIONS
                .get(duration)
                .ok_or_else(|| ActionError(format!("no duration {duration}")))?
        } else {
            0
        };
        Action::new(kind, d, u)
    }

    /// (kind, direction, duration) head indices; non-applicable heads are 0.
    pub fn to_indices(&self) -> [usize; 3] {
        [
            self.kind.index(),
            if is_directional(self.kind) {
                direction_index(self.direction).unwrap_or(0)
            } else {
                0
            },
            if is_timed(self.kind) {
                duration_index(self.duration).unwrap_or(0)
            } else {
                0
            },
        ]
    }

    /// Parses 'MOVE N 16', 'sword e', 'wait 8', 'boomerang nw' (timed kinds default to 16).
    pub fn parse(text: &str) -> Result<Action, ActionError> {
        let upper = text.to_uppercase();
        let mut parts = upper.split_whitespace();
        let kind_name = parts
            .next()
            .ok_or_else(|| ActionError("empty action".into()))?;
        let kind = ActionKind::from_name(kind_name)
            .ok_or_else(|| ActionError(format!("unknown kind {kind_name}")))?;
        let mut direction = Direction::None;
        let mut duration = 0;
        if is_directional(kind) {
            let d = parts
                .next()
                .ok_or_else(|| ActionError(format!("{} requires a direction", kind.name())))?;
            direction = Direction::from_name(d)
                .ok_or_else(|| ActionError(format!("unknown direction {d}")))?;
        }
        if is_timed(kind) {
            duration = match parts.next() {
                Some(d) => d
                    .parse()
                    .map_err(|_| ActionError(format!("bad duration {d}")))?,
                None => 16,
            };
        }
        let rest: Vec<&str> = parts.collect();
        if !rest.is_empty() {
            return Err(ActionError(format!(
                "Unexpected tokens in action '{text}': {rest:?}"
            )));
        }
        Action::new(kind, direction, duration)
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind.name())?;
        if is_directional(self.kind) {
            write!(f, " {}", self.direction.name())?;
        }
        if is_timed(self.kind) {
            write!(f, " {}", self.duration)?;
        }
        Ok(())
    }
}

/// Legal actions: kind[K], direction[K][D], duration[K][U].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActionMask {
    pub kind: [bool; NUM_KINDS],
    pub direction: [[bool; NUM_DIRECTIONS]; NUM_KINDS],
    pub duration: [[bool; NUM_DURATIONS]; NUM_KINDS],
}

impl ActionMask {
    /// Allows `kind` (with `direction`); timed kinds get every duration.
    pub fn allow(&mut self, kind: ActionKind, direction: Option<Direction>) {
        let k = kind.index();
        self.kind[k] = true;
        if let Some(d) = direction {
            self.direction[k][direction_index(d).expect("mask direction")] = true;
        }
        if is_timed(kind) {
            self.duration[k] = [true; NUM_DURATIONS];
        }
    }

    pub fn disallow_kind(&mut self, kind: ActionKind) {
        let k = kind.index();
        self.kind[k] = false;
        self.direction[k] = [false; NUM_DIRECTIONS];
        self.duration[k] = [false; NUM_DURATIONS];
    }

    pub fn is_allowed(&self, action: &Action) -> bool {
        let k = action.kind.index();
        if !self.kind[k] {
            return false;
        }
        if is_directional(action.kind)
            && !direction_index(action.direction).is_some_and(|i| self.direction[k][i])
        {
            return false;
        }
        if is_timed(action.kind)
            && !duration_index(action.duration).is_some_and(|i| self.duration[k][i])
        {
            return false;
        }
        true
    }

    pub fn directions(&self, kind: ActionKind) -> Vec<Direction> {
        DIRECTIONS
            .iter()
            .enumerate()
            .filter(|(i, _)| self.direction[kind.index()][*i])
            .map(|(_, d)| *d)
            .collect()
    }

    pub fn to_flat(&self) -> [bool; MASK_SIZE] {
        let mut out = [false; MASK_SIZE];
        out[..NUM_KINDS].copy_from_slice(&self.kind);
        for k in 0..NUM_KINDS {
            out[NUM_KINDS + k * NUM_DIRECTIONS..NUM_KINDS + (k + 1) * NUM_DIRECTIONS]
                .copy_from_slice(&self.direction[k]);
            let base = NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS + k * NUM_DURATIONS;
            out[base..base + NUM_DURATIONS].copy_from_slice(&self.duration[k]);
        }
        out
    }

    pub fn from_flat(flat: &[bool; MASK_SIZE]) -> ActionMask {
        let mut m = ActionMask::default();
        m.kind.copy_from_slice(&flat[..NUM_KINDS]);
        for k in 0..NUM_KINDS {
            m.direction[k].copy_from_slice(
                &flat[NUM_KINDS + k * NUM_DIRECTIONS..NUM_KINDS + (k + 1) * NUM_DIRECTIONS],
            );
            let base = NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS + k * NUM_DURATIONS;
            m.duration[k].copy_from_slice(&flat[base..base + NUM_DURATIONS]);
        }
        m
    }
}

impl fmt::Display for ActionMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        for kind in ActionKind::ALL {
            if !self.kind[kind.index()] {
                continue;
            }
            let mut text = kind.name().to_owned();
            if is_directional(kind) {
                let dirs: Vec<&str> = self.directions(kind).iter().map(|d| d.name()).collect();
                text += &format!("[{}]", dirs.join(","));
            }
            parts.push(text);
        }
        f.write_str(&parts.join(" "))
    }
}

/// Legal actions of a state (`compute_action_mask`). An attack is legal only where the game would start its weapon:
/// [`GameView::weapon_ready`] and the facing inside Link_FilterInput's BorderBounds (`sword_directions`,
/// `item_directions`; a diagonal throw faces N or S).
pub fn compute_action_mask(state: &GameView) -> ActionMask {
    let mut mask = ActionMask::default();
    let available = state.available_actions();

    for d in CARDINALS {
        if state.can_link_move(d) || state.is_push_block_direction(d) {
            mask.allow(ActionKind::Move, Some(d));
        }
    }
    mask.allow(ActionKind::Wait, None);

    let has_enemies = state.has_active_enemies();
    let has_items = !state.items.is_empty();
    let has_bomb_walls = !state.bomb_walls().is_empty() || state.ow_bomb_secret_closed();
    // In a doorway ($53) the game never turns Link sideways: no attack across the doorway's axis.
    let doorway = Direction::from_value(state.var(crate::vars::Var::DoorwayDir) as u8)
        .filter(|_| state.var(crate::vars::Var::DoorwayDir) != 0);
    let across = |d: Direction| {
        doorway.is_some_and(|dw| d != dw && d != dw.opposite())
    };
    let sword_dirs: Vec<Direction> = state.sword_directions().into_iter().filter(|&d| !across(d)).collect();
    let item_dirs: Vec<Direction> = state.item_directions().into_iter().filter(|&d| !across(d)).collect();

    for kind in [
        ActionKind::Sword,
        ActionKind::Beams,
        ActionKind::Boomerang,
        ActionKind::Arrow,
        ActionKind::Bomb,
    ] {
        if !available[kind.index()] || !state.weapon_ready(kind) {
            continue;
        }
        let pickup = matches!(
            kind,
            ActionKind::Sword | ActionKind::Beams | ActionKind::Boomerang | ActionKind::Arrow
        );
        if kind == ActionKind::Bomb {
            if !(has_enemies || has_bomb_walls) {
                continue;
            }
        } else if !has_enemies && !(has_items && pickup) {
            continue;
        }
        let dirs = if matches!(kind, ActionKind::Sword | ActionKind::Beams) {
            &sword_dirs
        } else {
            &item_dirs
        };
        for d in CARDINALS {
            if dirs.contains(&d) {
                mask.allow(kind, Some(d));
            }
        }
        if is_diagonal(kind) {
            for (diag, facing) in [
                (Direction::NW, Direction::N),
                (Direction::NE, Direction::N),
                (Direction::SW, Direction::S),
                (Direction::SE, Direction::S),
            ] {
                if item_dirs.contains(&facing) {
                    mask.allow(kind, Some(diag));
                }
            }
        }
    }

    if !mask.direction[ActionKind::Move.index()].iter().any(|&b| b) {
        for d in CARDINALS {
            mask.allow(ActionKind::Move, Some(d));
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    //! Port of the pure action tests in tests/test_core.py.
    use super::*;

    #[test]
    fn action_indices_round_trip() {
        for text in [
            "MOVE N 16",
            "WAIT 8",
            "SWORD E",
            "BOOMERANG NW",
            "ARROW S",
            "BOMB W",
            "BEAMS N",
        ] {
            let a = Action::parse(text).unwrap();
            let [k, d, u] = a.to_indices();
            assert_eq!(Action::from_indices(k, d, u).unwrap(), a, "{text}");
            assert_eq!(a.to_string(), text);
        }
    }

    #[test]
    fn from_indices_ignores_heads_that_do_not_apply() {
        assert_eq!(
            Action::from_indices(1, 5, 2).unwrap(),
            Action::parse("WAIT 16").unwrap()
        );
        assert_eq!(
            Action::from_indices(2, 3, 3).unwrap(),
            Action::parse("SWORD E").unwrap()
        );
    }

    #[test]
    fn invalid_actions_are_rejected() {
        for text in ["SWORD NW", "MOVE NE 8", "MOVE N 5", "WAIT 3", "BOMB"] {
            assert!(Action::parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn mask_flat_round_trip() {
        let mut mask = ActionMask::default();
        mask.allow(ActionKind::Move, Some(Direction::N));
        mask.allow(ActionKind::Wait, None);
        mask.allow(ActionKind::Boomerang, Some(Direction::SE));
        let flat = mask.to_flat();
        assert_eq!(flat.len(), 91);
        assert_eq!(ActionMask::from_flat(&flat), mask);
        assert!(mask.is_allowed(&Action::parse("MOVE N 32").unwrap()));
        assert!(!mask.is_allowed(&Action::parse("MOVE S 32").unwrap()));
        assert!(mask.is_allowed(&Action::parse("BOOMERANG SE").unwrap()));
        mask.disallow_kind(ActionKind::Boomerang);
        assert!(!mask.is_allowed(&Action::parse("BOOMERANG SE").unwrap()));
        assert_eq!(mask.to_string(), "MOVE[N] WAIT");
    }
}
