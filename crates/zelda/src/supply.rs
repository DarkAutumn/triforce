//! Health supply through one room visit (round 25): kills and the damage-free kill streak (`$627`, the fairy at 16:
//! asm:Damage_KillMonster, asm:Drop_SetUpItem), and the fate of every dropped heart and fairy with what Link faced
//! when it fell (how hurt he was, how far the drop and the objective were). Rules: docs/architecture.md "Supply".

use crate::change::Change;
use crate::enums::item;
use crate::game::GameView;
use crate::hearts::TIMER_MARGIN;
use crate::linger::linger_wavefront;
use crate::objectives::Objective;
use crate::room::Target;

/// What became of a dropped heart or fairy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropFate {
    Taken,
    Expired,
    /// Still on the floor when Link left the room.
    Left,
    /// Still on the floor when Link died.
    Death,
}

impl DropFate {
    pub fn name(self) -> &'static str {
        match self {
            DropFate::Taken => "taken",
            DropFate::Expired => "expired",
            DropFate::Left => "left",
            DropFate::Death => "death",
        }
    }
}

/// One dropped heart or fairy.
#[derive(Clone, Debug, PartialEq)]
pub struct DropNote {
    pub fairy: bool,
    /// Actions into the visit when it appeared.
    pub step: u32,
    /// Hearts Link was missing then (max − health).
    pub deficit: f64,
    /// Link's walking distance in tiles to the drop and to the objective's targets then (None: unreachable).
    pub to_drop: Option<i32>,
    pub to_objective: Option<i32>,
    pub fate: DropFate,
    /// Actions from its appearance to its fate.
    pub after: u32,
    /// Actions that cost Link health in that time.
    pub hits: u32,
}

/// The visit's kills and kill streak.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SupplyStats {
    /// Kills (`$627` increments; one that a hit in the same step cleared counts what remains).
    pub kills: u32,
    /// `$627` on entering the room, and its highest value in the visit.
    pub streak_in: u32,
    pub streak_max: u32,
    pub fairies_dropped: u32,
    pub fairies_taken: u32,
}

#[derive(Clone, Debug)]
struct Open {
    slot: i8,
    id: i16,
    timer: u8,
    note: usize,
}

/// Follows one room visit (same lifecycle as [`crate::hearts::HeartTracker`]).
#[derive(Clone, Debug, Default)]
pub struct SupplyTracker {
    pub stats: SupplyStats,
    pub drops: Vec<DropNote>,
    open: Vec<Open>,
    location: Option<crate::enums::MapLocation>,
    steps: u32,
}

fn heals(id: i16) -> bool {
    id == item::HEART || id == item::FAIRY
}

impl SupplyTracker {
    /// Starts a visit in `state`'s room; drops already on the floor are not followed.
    pub fn reset(&mut self, state: &GameView) {
        let streak = state.kill_streak().max(0) as u32;
        *self = SupplyTracker {
            stats: SupplyStats {
                streak_in: streak,
                streak_max: streak,
                ..SupplyStats::default()
            },
            location: Some(state.full_location()),
            ..SupplyTracker::default()
        };
    }

    /// Follows one step of the visit; `state` is the state after it (cheats applied), `objective` its route
    /// objective (for the distance to the objective when a drop appears).
    pub fn update(&mut self, change: &Change, state: &GameView, objective: &Objective) {
        if self.location.is_none() {
            return;
        }
        self.steps += 1;
        let (prev, curr) = (change.previous.kill_streak().max(0), state.kill_streak().max(0));
        self.stats.kills += if curr >= prev { curr - prev } else { curr } as u32;
        self.stats.streak_max = self.stats.streak_max.max(curr as u32);
        if state.game_over() || Some(state.full_location()) != self.location || change.maze_loop().is_some() {
            return;
        }
        let hit = change.health_lost > 0.0;
        let reach = change.frames / 2 + TIMER_MARGIN;
        let gained = change.health_gained > 0.0;
        let mut still = Vec::with_capacity(self.open.len());
        for open in self.open.drain(..) {
            let now = state.items.iter().find(|i| i.index == open.slot && i.id == open.id);
            let note = &mut self.drops[open.note];
            match now {
                Some(i) if i.timer <= open.timer => {
                    note.after += 1;
                    note.hits += u32::from(hit);
                    still.push(Open { timer: i.timer, ..open });
                }
                _ => {
                    note.after += 1;
                    note.hits += u32::from(hit);
                    let taken = open.timer as u32 > reach || gained;
                    note.fate = if taken { DropFate::Taken } else { DropFate::Expired };
                    if taken && open.id == item::FAIRY {
                        self.stats.fairies_taken += 1;
                    }
                }
            }
        }
        self.open = still;
        for it in state.items.iter().filter(|i| heals(i.id)) {
            if self.open.iter().any(|o| o.slot == it.index) {
                continue;
            }
            let was = change.previous.items.iter().any(|p| p.index == it.index && p.id == it.id && p.timer >= it.timer);
            if was {
                continue;
            }
            let fairy = it.id == item::FAIRY;
            self.stats.fairies_dropped += u32::from(fairy);
            let link = state.link.tile();
            let to_objective = linger_wavefront(state, objective, None).get(link);
            let mut toward = objective.clone();
            toward.pbrs_targets = it.link_overlap_tiles().into_iter().map(Target::Tile).collect();
            let to_drop = linger_wavefront(state, &toward, None).get(link);
            self.drops.push(DropNote {
                fairy,
                step: self.steps,
                deficit: state.max_health() as f64 - state.health(),
                to_drop,
                to_objective,
                fate: DropFate::Left,
                after: 0,
                hits: 0,
            });
            self.open.push(Open {
                slot: it.index,
                id: it.id,
                timer: it.timer,
                note: self.drops.len() - 1,
            });
        }
    }

    /// Ends the visit: drops still followed are left (or lost with Link, `died`). Returns the visit's stats and
    /// drops, once (None when no visit is open).
    pub fn finish(&mut self, died: bool) -> Option<(SupplyStats, Vec<DropNote>)> {
        self.location?;
        let fate = if died { DropFate::Death } else { DropFate::Left };
        for open in self.open.drain(..) {
            self.drops[open.note].fate = fate;
        }
        self.location = None;
        Some((self.stats, std::mem::take(&mut self.drops)))
    }
}
