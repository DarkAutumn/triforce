//! The streams read with the names their replay's META gives them (`tables`, `actions`): what a viewer, `tf replay
//! targets` and the Python reader print, with nothing but the replay in hand.

use serde_json::{Map, Value, json};

use crate::format::{Decision, Mask, RoomRef, TARGET_OBS, TARGET_REWARD_ONLY, TargetRef, Targets, legal_joint};

/// A replay's name tables (META `tables` and `actions`).
#[derive(Debug, Clone, Default)]
pub struct Names {
    objective_kinds: Vec<String>,
    roles: Vec<(String, Vec<String>)>,
    treasures: Vec<String>,
    routes: Vec<String>,
    stretches: Vec<String>,
    mask_reasons: Vec<String>,
    draws: Vec<String>,
    kinds: Vec<String>,
    directions: Vec<String>,
    durations: Vec<u64>,
    directional: Vec<bool>,
    timed: Vec<bool>,
}

/// One legal joint action of a decision ([`Names::choices`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// Its joint index (`(kind * directions + direction) * durations + duration`).
    pub joint: usize,
    pub kind: usize,
    pub direction: usize,
    pub duration: usize,
    /// For people: `MOVE N 16`, `SWORD E`, `WAIT 8`, `RECORDER` ([`Names::action_text`]).
    pub label: String,
    /// Its probability under the network's distribution.
    pub p: f32,
    /// P(Link is hit) if taken, where the network has a hit head.
    pub danger: Option<f32>,
}

fn strings(value: Option<&Value>) -> Vec<String> {
    value.and_then(Value::as_array).map_or_else(Vec::new, |a| {
        a.iter().map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned)).collect()
    })
}

fn bools(value: Option<&Value>) -> Vec<bool> {
    value.and_then(Value::as_array).map_or_else(Vec::new, |a| a.iter().filter_map(Value::as_bool).collect())
}

fn name(names: &[String], index: usize) -> String {
    names.get(index).cloned().unwrap_or_else(|| format!("#{index}"))
}

/// `8:0x4c`, a cave or cellar `0:0x77:cave`.
pub fn room_text(r: &RoomRef) -> String {
    format!("{}:0x{:02x}{}", r.level, r.room, if r.in_cave { ":cave" } else { "" })
}

impl Names {
    pub fn of(meta: &Map<String, Value>) -> Names {
        let tables = meta.get("tables");
        let table = |key: &str| strings(tables.and_then(|t| t.get(key)));
        let actions = meta.get("actions");
        let roles = tables
            .and_then(|t| t.get("target_roles"))
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |roles| {
                roles
                    .iter()
                    .map(|r| {
                        let name = r.get("name").and_then(Value::as_str).unwrap_or("?").to_owned();
                        (name, strings(r.get("fields")))
                    })
                    .collect()
            });
        Names {
            objective_kinds: table("objective_kinds"),
            roles,
            treasures: table("treasures"),
            routes: table("routes"),
            stretches: table("stretches"),
            mask_reasons: table("mask_reasons"),
            draws: table("draws"),
            kinds: strings(actions.and_then(|a| a.get("kinds"))),
            directions: strings(actions.and_then(|a| a.get("directions"))),
            durations: actions
                .and_then(|a| a.get("durations"))
                .and_then(Value::as_array)
                .map_or_else(Vec::new, |d| d.iter().filter_map(Value::as_u64).collect()),
            directional: bools(actions.and_then(|a| a.get("directional"))),
            timed: bools(actions.and_then(|a| a.get("timed"))),
        }
    }

    /// Whether kind `k` has a direction (META `actions.directional`; a replay that doesn't say: yes).
    pub fn directional(&self, k: usize) -> bool {
        self.directional.get(k).copied().unwrap_or(true)
    }

    /// Whether kind `k` has a duration (META `actions.timed`; a replay that doesn't say: yes).
    pub fn timed(&self, k: usize) -> bool {
        self.timed.get(k).copied().unwrap_or(true)
    }

    /// An action for people, with only the heads its kind has: `MOVE N 16`, `SWORD E`, `WAIT 8`, `RECORDER`.
    pub fn action_text(&self, kind: usize, direction: usize, duration: usize) -> String {
        let mut text = self.kind(kind);
        if self.directional(kind) {
            text += &format!(" {}", self.direction(direction));
        }
        if self.timed(kind) {
            text += &format!(" {}", self.duration(duration).unwrap_or(duration as u64));
        }
        text
    }

    /// A joint action index (`(kind * directions + direction) * durations + duration`) for people.
    pub fn joint_text(&self, joint: usize) -> String {
        let (dirs, durs) = (self.directions.len().max(1), self.durations.len().max(1));
        self.action_text(joint / (dirs * durs), (joint / durs) % dirs, joint % durs)
    }

    pub fn kind(&self, index: usize) -> String {
        name(&self.kinds, index)
    }

    pub fn direction(&self, index: usize) -> String {
        name(&self.directions, index)
    }

    /// Duration `index`'s frames.
    pub fn duration(&self, index: usize) -> Option<u64> {
        self.durations.get(index).copied()
    }

    pub fn draw(&self, index: usize) -> String {
        name(&self.draws, index)
    }

    pub fn objective(&self, index: usize) -> String {
        name(&self.objective_kinds, index)
    }

    /// A target's role name and its fields by name (a direction by its name, a treasure's item by its name, tiles'
    /// columns and rows as signed numbers).
    pub fn target(&self, t: &TargetRef) -> (String, Map<String, Value>) {
        let (role, fields) = self
            .roles
            .get(usize::from(t.role))
            .cloned()
            .unwrap_or_else(|| (format!("#{}", t.role), Vec::new()));
        let mut out = Map::new();
        for (field, &byte) in fields.iter().zip(&t.fields) {
            let value = match field.as_str() {
                "dir" => json!(self.direction(usize::from(byte))),
                "item" if role == "treasure" => json!(name(&self.treasures, usize::from(byte))),
                "col" | "row" => json!(byte as i8),
                _ => json!(byte),
            };
            out.insert(field.clone(), value);
        }
        (role, out)
    }

    /// One target for people: `bomb-wall N (col 15, row 2)`, `monster slot 3 type 0x2a`, `treasure key`.
    pub fn target_text(&self, t: &TargetRef) -> String {
        let (role, fields) = self.target(t);
        let mut text = role.clone();
        let tile = matches!((fields.get("col"), fields.get("row")), (Some(_), Some(_)));
        if let Some(Value::String(d)) = fields.get("dir") {
            text += &format!(" {d}");
        }
        match role.as_str() {
            "monster" => text += &format!(" slot {} type 0x{:02x}", fields["slot"], fields["type"].as_u64().unwrap_or(0)),
            "item" => text += &format!(" slot {} item 0x{:02x}", fields["slot"], fields["item"].as_u64().unwrap_or(0)),
            "treasure" => text += &format!(" {}", fields["item"].as_str().unwrap_or("?")),
            "whirlwind" => text += &format!(" row {}", fields["row"]),
            _ => {}
        }
        if tile {
            text += &format!(" ({}, {})", fields["col"], fields["row"]);
        }
        if t.flags & TARGET_REWARD_ONLY != 0 {
            text += " [reward-only]";
        }
        text
    }

    /// A TARGETS record as a JSON object of names: `frame`, `objective`, `flags`, `route`, `step`, `way`,
    /// `stretch`, `next_rooms`, `retreats`, `targets` (each `{role, obs, reward_only, <fields>, text}`, `text`:
    /// [`Names::target_text`]).
    pub fn targets_json(&self, r: &Targets) -> Value {
        let flags: Vec<&str> = ["treasure-refused", "drops-first", "bag-drops", "clips"]
            .iter()
            .enumerate()
            .filter(|(i, _)| r.flags & (1 << i) != 0)
            .map(|(_, f)| *f)
            .collect();
        let targets: Vec<Value> = r
            .targets
            .iter()
            .map(|t| {
                let (role, mut fields) = self.target(t);
                fields.insert("role".into(), json!(role));
                fields.insert("obs".into(), json!(t.flags & TARGET_OBS != 0));
                fields.insert("reward_only".into(), json!(t.flags & TARGET_REWARD_ONLY != 0));
                fields.insert("text".into(), json!(self.target_text(t)));
                Value::Object(fields)
            })
            .collect();
        json!({
            "frame": r.frame,
            "objective": self.objective(usize::from(r.objective)),
            "flags": flags,
            "route": r.route.map(|i| name(&self.routes, usize::from(i))),
            "step": r.step,
            "way": ([None, Some("clips"), Some("plain")].get(usize::from(r.way)).copied().flatten()),
            "stretch": r.stretch.map(|i| name(&self.stretches, usize::from(i))),
            "next_rooms": r.next_rooms.iter().map(room_text).collect::<Vec<_>>(),
            "retreats": r.retreats.iter().map(room_text).collect::<Vec<_>>(),
            "targets": targets,
        })
    }

    /// A MASKS record as a JSON object of names: `frame`, `kinds` ({kind: {directions, durations}}: a kind without
    /// a direction or a duration head lists none), `blocked` ({kind: [reasons]}), `text` ([`Names::mask_text`]).
    pub fn mask_json(&self, m: &Mask) -> Value {
        let mut kinds = Map::new();
        for k in (0..16).filter(|&k| m.allows(k)) {
            let directions: Vec<String> = (0..8)
                .filter(|d| self.directional(k) && m.directions[k] & (1 << d) != 0)
                .map(|d| self.direction(d))
                .collect();
            let durations: Vec<u64> = (0..8)
                .filter(|u| self.timed(k) && m.durations[k] & (1 << u) != 0)
                .map(|u| self.durations.get(u).copied().unwrap_or(u as u64))
                .collect();
            kinds.insert(self.kind(k), json!({"directions": directions, "durations": durations}));
        }
        let mut blocked = Map::new();
        for &(kind, reason) in &m.blocked {
            let entry = blocked.entry(self.kind(usize::from(kind))).or_insert_with(|| json!([]));
            entry.as_array_mut().expect("array").push(json!(name(&self.mask_reasons, usize::from(reason))));
        }
        json!({"frame": m.frame, "kinds": kinds, "blocked": blocked, "text": self.mask_text(m)})
    }

    /// A MASKS record for people: `MOVE[N,S,W,E] WAIT SWORD[N] BOMB blocked: tenth`.
    pub fn mask_text(&self, m: &Mask) -> String {
        let mut parts: Vec<String> = Vec::new();
        for k in (0..16).filter(|&k| m.allows(k)) {
            let dirs: Vec<String> = (0..8)
                .filter(|d| self.directional(k) && m.directions[k] & (1 << d) != 0)
                .map(|d| self.direction(d))
                .collect();
            parts.push(if dirs.is_empty() { self.kind(k) } else { format!("{}[{}]", self.kind(k), dirs.join(",")) });
        }
        for &(kind, reason) in &m.blocked {
            let state = if m.allows(usize::from(kind)) { "kept" } else { "blocked" };
            parts.push(format!(
                "{} {state}: {}",
                self.kind(usize::from(kind)),
                name(&self.mask_reasons, usize::from(reason))
            ));
        }
        parts.join(" ")
    }

    /// A decision's legal joint actions with their probabilities (and P(hit) where the network has a hit head),
    /// most likely first.
    pub fn choices(&self, d: &Decision<'_>) -> Vec<Choice> {
        let dist: Vec<f32> = d.distribution().collect();
        let (dirs, durs) = (self.directions.len().max(1), self.durations.len().max(1));
        let kinds = dist.len() / (1 + dirs + durs);
        let dangers = d.dangers();
        let mut out: Vec<Choice> = legal_joint(dist.len(), |i| dist[i].is_finite())
            .map(|j| {
                let joint = usize::from(j);
                let (kind, direction, duration) = (joint / (dirs * durs), (joint / durs) % dirs, joint % durs);
                let lp = dist[kind]
                    + dist[kinds + kind * dirs + direction]
                    + dist[kinds + kinds * dirs + kind * durs + duration];
                let danger = dangers.iter().find(|&&(at, _)| usize::from(at) == joint).map(|&(_, p)| p);
                let label = self.action_text(kind, direction, duration);
                Choice { joint, kind, direction, duration, label, p: lp.exp(), danger }
            })
            .collect();
        out.sort_by(|a, b| b.p.total_cmp(&a.p));
        out
    }
}
