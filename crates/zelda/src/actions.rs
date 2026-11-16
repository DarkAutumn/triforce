//! The v2 action schema and masking (`triforce/actions.py`): kind x direction x duration timed options.

use crate::enums::{ActionKind, Direction};
use crate::game::GameView;
use std::fmt;

/// 2: the CLIP kind (round 21, the grid-point skip macro). 3: RECORDER, FOOD and CANDLE (round 24; kinds 8-10, the
/// existing indices kept, so a schema 2 checkpoint is widened by `triforce convert-model`, not retrained).
pub const ACTION_SCHEMA_VERSION: u32 = 3;

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

pub const NUM_KINDS: usize = 11;
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

/// Every kind except WAIT and RECORDER has a direction.
pub const fn is_directional(kind: ActionKind) -> bool {
    !matches!(kind, ActionKind::Wait | ActionKind::Recorder)
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

    /// Removes `kind` in `direction`; the kind goes when no direction is left.
    pub fn disallow(&mut self, kind: ActionKind, direction: Direction) {
        let k = kind.index();
        if let Some(i) = direction_index(direction) {
            self.direction[k][i] = false;
        }
        if !self.direction[k].iter().any(|&b| b) {
            self.disallow_kind(kind);
        }
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

/// Legal actions of a state (`compute_action_mask`). An attack is legal only where the game would start its weapon
/// ([`GameView::weapon_ready`]) and the buttons can make that strike from here ([`crate::attack::plan`]: the press
/// or a turn first, inside asm:Link_FilterInput's border bands, no sideways turn the game refuses); a B item that
/// is not the selected one also needs the item sub-screen to open ([`crate::submenu::can_open`]). RECORDER, FOOD and
/// CANDLE follow the same rules (the item held, the game would use it, the buttons can); RECORDER only where
/// [`recorder_wanted`] (a route narrows it per step: `crate::Session::set_recorder_facings`), FOOD only where a
/// Grumble waits for it ([`food_wanted`]), and no BOOMERANG while placed food lies (round 26: the boomerang shares its
/// slot and would take it away, asm:Boomerang_Wield). MOVE also presses into a secret Armos from its side
/// ([`GameView::is_secret_armos_direction`]) and into the secret gravestone from below (round 27,
/// [`GameView::is_secret_push_direction`]). A family's `items:` switches mask them further
/// (`env::config::ItemsConfig`), and a route narrows ARROW and MOVE (`crate::Session::set_arrow_facings`,
/// `crate::Session::set_vetoed_doorways`).
pub fn compute_action_mask(state: &GameView) -> ActionMask {
    let mut mask = ActionMask::default();

    for d in CARDINALS {
        if state.can_link_move(d)
            || state.is_push_block_direction(d)
            || state.is_secret_armos_direction(d)
            || state.is_secret_push_direction(d)
        {
            mask.allow(ActionKind::Move, Some(d));
        }
    }
    mask.allow(ActionKind::Wait, None);

    let has_enemies = state.has_active_enemies();
    let has_items = !state.items.is_empty();
    let has_bomb_walls = !state.bomb_walls().is_empty() || state.ow_bomb_secret_closed();
    let menu_opens = crate::submenu::can_open(state);
    let available = state.available_actions();
    let ready = |kind: ActionKind| usable(state, &available, kind, menu_opens);

    for kind in [
        ActionKind::Sword,
        ActionKind::Beams,
        ActionKind::Boomerang,
        ActionKind::Arrow,
        ActionKind::Bomb,
    ] {
        if !ready(kind) || (kind == ActionKind::Boomerang && food_lies(state)) {
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
        for d in state.attack_directions(kind) {
            mask.allow(kind, Some(d));
        }
    }
    for kind in [ActionKind::Food, ActionKind::Candle] {
        if ready(kind) && (kind != ActionKind::Food || food_wanted(state)) {
            for d in state.attack_directions(kind) {
                mask.allow(kind, Some(d));
            }
        }
    }
    if recorder_wanted(state)
        && ready(ActionKind::Recorder)
        && crate::attack::plan(state, ActionKind::Recorder, Direction::None).is_some()
    {
        mask.allow(ActionKind::Recorder, None);
    }

    if !mask.direction[ActionKind::Move.index()].iter().any(|&b| b) {
        for d in CARDINALS {
            mask.allow(ActionKind::Move, Some(d));
        }
    }
    mask
}

/// The inventory holds `kind` (`available`, [`GameView::available_actions`]), the game would start it now and, for a B
/// item not selected, the sub-screen opens (`menu_opens`, [`crate::submenu::can_open`]).
fn usable(state: &GameView, available: &[bool; NUM_KINDS], kind: ActionKind, menu_opens: bool) -> bool {
    available[kind.index()]
        && state.weapon_ready(kind)
        && !(crate::submenu::needs_selection(state, kind) && !menu_opens)
}

/// The game would play the recorder now, and the buttons can from here (the game's rules only: where the route wants
/// it is [`recorder_wanted`]'s, or the route's).
pub fn recorder_playable(state: &GameView) -> bool {
    usable(state, &state.available_actions(), ActionKind::Recorder, crate::submenu::can_open(state))
        && crate::attack::plan(state, ActionKind::Recorder, Direction::None).is_some()
}

/// Where playing the recorder does something a dungeon route wants (round 24): a dungeon room in play (mode 5) that
/// holds a big Digdogger ($38, $39) and has not heard the recorder since Link came in (`FluteUsedState` $51B = 0; the
/// game zeroes it on room entry). There the tune starts the Digdogger's shrink and split (asm:Digdogger_HandleFlute,
/// zelda-asm/enemies.md "Digdogger"); a second play changes nothing. Without a route nowhere else: on the overworld
/// it summons the whirlwind (asm:Whirlwind_Summon) or drains the pond of Q1 room 0x42 (asm:Flute_SecretRoomIdsOW),
/// which only a route step wants (round 26: `crate::objectives::GameCompletion::recorder_facings`, which also keeps it
/// from a big Digdogger Link only crosses, 7:0x39).
pub fn recorder_wanted(state: &GameView) -> bool {
    use crate::enums::enemy::{DIGDOGGER2, DIGDOGGER3};
    state.level() != 0
        && state.peek(ADDR_MODE) == MODE_PLAY
        && state.peek(ADDR_FLUTE_USED_STATE) == 0
        && state.enemies.iter().any(|e| matches!(e.id, DIGDOGGER2 | DIGDOGGER3))
}

/// A Grumble Goriya waits for food (round 26, 7:0x28): slot 1 holds it (type `$36`) in its state 1, its text typed
/// (asm:Grumble_State1WaitForFood takes food lying anywhere in the room). Nowhere else is food wanted: placed, it
/// only lies there (asm:Food_Wield doesn't consume it) and keeps the boomerang's slot.
pub fn food_wanted(state: &GameView) -> bool {
    state.peek(ADDR_OBJ_TYPE + 1) == OBJ_GRUMBLE && state.peek(ADDR_OBJ_STATE + 1) == 1
}

/// Placed food lies in slot `$0F` (its state's bit 7, asm:Food_Wield).
pub fn food_lies(state: &GameView) -> bool {
    state.peek(ADDR_OBJ_STATE + 0x0F) & 0x80 != 0
}

const ADDR_MODE: u16 = 0x12;
const MODE_PLAY: u8 = 5;
/// asm:FluteUsedState: 0, 1 once the recorder was played in this room, 2 while a Digdogger shrinks.
const ADDR_FLUTE_USED_STATE: u16 = 0x51B;
/// `ObjType` and `ObjState` tables; the Grumble Goriya's type.
const ADDR_OBJ_TYPE: u16 = 0x34F;
const ADDR_OBJ_STATE: u16 = 0xAC;
const OBJ_GRUMBLE: u8 = 0x36;

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
            "CLIP W",
            "CLIP N",
            "BEAMS N",
            "RECORDER",
            "FOOD W",
            "CANDLE N",
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
        assert_eq!(Action::from_indices(8, 3, 2).unwrap(), Action::parse("RECORDER").unwrap());
    }

    #[test]
    fn invalid_actions_are_rejected() {
        for text in [
            "SWORD NW", "MOVE NE 8", "MOVE N 5", "WAIT 3", "BOMB", "CLIP", "CLIP SE", "CLIP W 8", "RECORDER N",
            "FOOD", "FOOD NE", "CANDLE", "CANDLE SW", "CANDLE E 8",
        ] {
            assert!(Action::parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn mask_flat_round_trip() {
        let mut mask = ActionMask::default();
        mask.allow(ActionKind::Move, Some(Direction::N));
        mask.allow(ActionKind::Wait, None);
        mask.allow(ActionKind::Boomerang, Some(Direction::SE));
        mask.allow(ActionKind::Recorder, None);
        let flat = mask.to_flat();
        assert_eq!(flat.len(), 11 + 11 * 8 + 11 * 4);
        assert_eq!(ActionMask::from_flat(&flat), mask);
        assert!(mask.is_allowed(&Action::parse("MOVE N 32").unwrap()));
        assert!(!mask.is_allowed(&Action::parse("MOVE S 32").unwrap()));
        assert!(mask.is_allowed(&Action::parse("BOOMERANG SE").unwrap()));
        assert!(mask.is_allowed(&Action::parse("RECORDER").unwrap()));
        assert!(!mask.is_allowed(&Action::parse("CANDLE N").unwrap()));
        mask.disallow_kind(ActionKind::Boomerang);
        assert!(!mask.is_allowed(&Action::parse("BOOMERANG SE").unwrap()));
        assert_eq!(mask.to_string(), "MOVE[N] WAIT RECORDER");
    }
}
