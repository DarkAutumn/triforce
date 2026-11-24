//! What a recording keeps of each decision besides the inputs (round 40, the owner: a replay alone says what Link
//! was steered to, what he could do and what the network thought): the objective and its targets with their roles
//! (TARGETS), the legal actions and the rules that blocked some (MASKS), and the decision itself (DECISIONS: the
//! quantized distribution, the act mode's draw, the hit head's danger, the foresight's vetoes). TARGETS and MASKS
//! are written when they change. The names they index are META `tables` ([`Tables::meta`]).

use super::ZeldaAction;
use super::nes::PoolNes;
use crate::pool::Decided;
use crate::rollout::Foreseen;
use env::config::{MASK_REASONS, RewardConfig};
use replay::{Mask, RoomRef, TargetRef, Targets, Thought};
use serde_json::{Value, json};
use zelda::actions::{ActionMask, NUM_DIRECTIONS, NUM_DURATIONS, direction_index, is_directional};
use zelda::game::GameView;
use zelda::objectives::{GameCompletion, Objective, ROLES, TREASURES};
use zelda::{ActionKind, GameMap, MapLocation, Nes, Session};

/// The objective kinds, by `zelda::objectives::ObjectiveKind`'s value.
const OBJECTIVE_KINDS: [&str; 6] = ["NONE", "MOVE", "CAVE", "TREASURE", "FIGHT", "ITEM"];

/// The name tables a recording's TARGETS, MASKS and DECISIONS index (META `tables`), from the game map: the same
/// data gives the same tables, so two replays of one build stitch.
#[derive(Clone, Debug)]
pub(super) struct Tables {
    routes: Vec<String>,
    stretches: Vec<String>,
    meta: Value,
}

impl Tables {
    pub fn of(map: &GameMap) -> Tables {
        let mut routes: Vec<String> = map.routes.keys().cloned().collect();
        routes.sort();
        let mut stretches: Vec<String> = map.stretches.values().flatten().map(|s| s.name.clone()).collect();
        stretches.sort();
        stretches.dedup();
        let roles: Vec<Value> = ROLES.iter().map(|(name, fields)| json!({"name": name, "fields": fields})).collect();
        let meta = json!({
            "objective_kinds": OBJECTIVE_KINDS,
            "target_roles": roles,
            "target_flags": {"obs": "in the target the observation's wavefront leads to",
                             "reward-only": "priced by the reward, not in the observation's target"},
            "treasures": &TREASURES[..],
            "routes": routes,
            "stretches": stretches,
            "ways": ["none", "clips", "plain"],
            "mask_reasons": MASK_REASONS,
            "draws": policy::act::DRAWS,
            "foresight_outcomes": ["", "safe", "replaced", "unsafe"],
            "fields": {"dir": "META actions.directions index", "col": "tile column (i8)", "row": "tile row (i8)",
                       "slot": "object slot", "type": "monster object type", "item": "item id (treasure: tables.treasures)"},
        });
        Tables { routes, stretches, meta }
    }

    pub fn meta(&self) -> &Value {
        &self.meta
    }
}

fn room(location: MapLocation) -> RoomRef {
    RoomRef { level: location.level, room: location.value, in_cave: location.in_cave }
}

fn direction(d: zelda::Direction) -> u8 {
    direction_index(d).map_or(0xFF, |i| i as u8)
}

/// The TARGETS record of `objective` in `state`, the acting family's `reward` saying which targets its observation
/// sees, `completion` labelling the route step. The targets are a set, written in (role, fields, flags) order: the
/// objective's labels come in its targets' tile order, so in a fight the order followed the monsters as they moved
/// and each change was a record (8:0x5e: 30 records for five keese and a drop).
fn targets(
    tables: &Tables,
    objective: &Objective,
    completion: Option<&GameCompletion>,
    state: &GameView,
    reward: &RewardConfig,
) -> Targets {
    let flags = [objective.treasure_refused, objective.drops_first, objective.bag_drops, objective.clips]
        .iter()
        .enumerate()
        .fold(0u8, |bits, (i, &on)| bits | u8::from(on) << i);
    let index = |names: &[String], name: Option<&str>| name.and_then(|n| names.iter().position(|x| x == n)).map(|i| i as u8);
    let mut recorded: Vec<TargetRef> = env::objective::recorded_targets(objective, state, reward)
        .into_iter()
        .map(|(role, obs)| TargetRef {
            role: role.index(),
            flags: if obs { replay::TARGET_OBS } else { replay::TARGET_REWARD_ONLY },
            fields: role.fields(direction),
        })
        .collect();
    recorded.sort_unstable_by_key(|t| (t.role, t.fields, t.flags));
    Targets {
        frame: 0,
        objective: objective.kind as u8,
        flags,
        route: index(&tables.routes, completion.map(GameCompletion::route)),
        step: completion.and_then(GameCompletion::step).and_then(|s| u16::try_from(s).ok()),
        way: completion.and_then(GameCompletion::step_way).map_or(0, |clips| if clips { 1 } else { 2 }),
        stretch: index(&tables.stretches, completion.and_then(GameCompletion::stretch)),
        next_rooms: objective.next_rooms.iter().copied().map(room).collect(),
        retreats: objective.retreats.iter().copied().map(room).collect(),
        targets: recorded,
    }
}

/// The MASKS record of `mask` and the env's `rules`: a rule is kept where it blocks its kind, or keeps a kind with
/// directions to some of them.
fn mask(mask: &ActionMask, rules: &[(ActionKind, &'static str)]) -> Mask {
    let mut out = Mask::default();
    let bits = |row: &[bool]| row.iter().enumerate().fold(0u8, |b, (i, &on)| b | u8::from(on) << i);
    for k in (0..mask.kind.len()).filter(|&k| mask.kind[k]) {
        out.kinds |= 1 << k;
        out.directions[k] = bits(&mask.direction[k]);
        out.durations[k] = bits(&mask.duration[k]);
    }
    for &(kind, why) in rules {
        let k = kind.index();
        let limits = is_directional(kind) && matches!(why, "bomb-targets" | "facings" | "doorways" | "bomb-catch");
        if (!mask.kind[k] || limits)
            && let Some(reason) = MASK_REASONS.iter().position(|r| *r == why)
        {
            out.blocked.push((k as u8, reason as u8));
        }
    }
    out
}

fn joint([kind, direction, duration]: ZeldaAction) -> u16 {
    ((kind * NUM_DIRECTIONS as i64 + direction) * NUM_DURATIONS as i64 + duration) as u16
}

/// What the env knows of a decision about to execute, for its recording.
pub(super) struct Situation<'a> {
    pub objective: Option<&'a Objective>,
    pub completion: Option<&'a GameCompletion>,
    pub reward: Option<&'a RewardConfig>,
    pub rules: &'a [(ActionKind, &'static str)],
}

/// What a decision's recording holds besides the decision: the objective's record (None without one) and the mask's.
pub(super) struct Seen(Option<Targets>, Mask);

/// The objective and mask records of the decision about to execute in `session`, None when it isn't recording.
pub(super) fn observe<N: Nes>(tables: &Tables, at: Situation<'_>, session: &Session<PoolNes<N>>) -> Option<Seen> {
    if session.nes().recorder().is_none() {
        return None;
    }
    let state = session.try_state()?;
    let targets = at.objective.zip(at.reward).map(|(o, r)| targets(tables, o, at.completion, state, r));
    Some(Seen(targets, mask(session.mask(), at.rules)))
}

/// Records decision `d` in `session`'s recording: the objective and the mask when they changed (`seen`), then the
/// decision.
pub(super) fn write<N: Nes>(session: &mut Session<PoolNes<N>>, Seen(targets, mask): Seen, d: &Decided<'_, ZeldaAction>) {
    let Some(rec) = session.nes_mut().recorder_mut() else {
        return;
    };
    if let Some(targets) = &targets {
        rec.targets(targets);
    }
    rec.mask(&mask);
    let Some(dist) = d.distribution else {
        return;
    };
    let vetoed: Vec<u16> = d.vetoed.iter().map(|&a| joint(a)).collect();
    let foresight = d.decision.foresee.map(|f| {
        let outcome = match f.outcome {
            Foreseen::Safe => 1,
            Foreseen::Replaced => 2,
            Foreseen::Unsafe => 3,
        };
        (outcome, f.trials, f.frames)
    });
    rec.decision(&Thought {
        policy: d.policy as u8,
        action: d.action.map(|i| i as u8),
        value: d.decision.value,
        draw: d.decision.draw,
        low: d.decision.low,
        dist,
        danger: d.danger,
        foresight: foresight.map(|(outcome, trials, frames)| (outcome, trials, frames, &vetoed[..])),
    });
}
