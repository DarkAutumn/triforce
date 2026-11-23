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
}

fn strings(value: Option<&Value>) -> Vec<String> {
    value.and_then(Value::as_array).map_or_else(Vec::new, |a| {
        a.iter().map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned)).collect()
    })
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
        }
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
    /// `stretch`, `next_rooms`, `retreats`, `targets` (each `{role, obs, reward_only, <fields>}`).
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

    /// A MASKS record as a JSON object of names: `frame`, `kinds` ({kind: {directions, durations}}), `blocked`
    /// ({kind: [reasons]}).
    pub fn mask_json(&self, m: &Mask) -> Value {
        let mut kinds = Map::new();
        for k in (0..16).filter(|&k| m.allows(k)) {
            let directions: Vec<String> =
                (0..8).filter(|d| m.directions[k] & (1 << d) != 0).map(|d| self.direction(d)).collect();
            let durations: Vec<u64> = (0..8)
                .filter(|u| m.durations[k] & (1 << u) != 0)
                .map(|u| self.durations.get(u).copied().unwrap_or(u as u64))
                .collect();
            kinds.insert(self.kind(k), json!({"directions": directions, "durations": durations}));
        }
        let mut blocked = Map::new();
        for &(kind, reason) in &m.blocked {
            let entry = blocked.entry(self.kind(usize::from(kind))).or_insert_with(|| json!([]));
            entry.as_array_mut().expect("array").push(json!(name(&self.mask_reasons, usize::from(reason))));
        }
        json!({"frame": m.frame, "kinds": kinds, "blocked": blocked})
    }

    /// A MASKS record for people: `MOVE[N,S,W,E] SWORD[N] BOMB blocked: tenth`.
    pub fn mask_text(&self, m: &Mask) -> String {
        let mut parts: Vec<String> = Vec::new();
        for k in (0..16).filter(|&k| m.allows(k)) {
            let dirs: Vec<String> =
                (0..8).filter(|d| m.directions[k] & (1 << d) != 0).map(|d| self.direction(d)).collect();
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
    /// most likely first: `(kind, direction, duration, p, danger)`.
    pub fn choices(&self, d: &Decision<'_>) -> Vec<(String, String, u64, f32, Option<f32>)> {
        let dist: Vec<f32> = d.distribution().collect();
        let (dirs, durs) = (self.directions.len().max(1), self.durations.len().max(1));
        let kinds = dist.len() / (1 + dirs + durs);
        let dangers = d.dangers();
        let mut out: Vec<_> = legal_joint(dist.len(), |i| dist[i].is_finite())
            .map(|j| {
                let j = usize::from(j);
                let (k, di, u) = (j / (dirs * durs), (j / durs) % dirs, j % durs);
                let lp = dist[k] + dist[kinds + k * dirs + di] + dist[kinds + kinds * dirs + k * durs + u];
                let danger = dangers.iter().find(|&&(at, _)| usize::from(at) == j).map(|&(_, p)| p);
                (self.kind(k), self.direction(di), self.durations.get(u).copied().unwrap_or(0), lp.exp(), danger)
            })
            .collect();
        out.sort_by(|a, b| b.3.total_cmp(&a.3));
        out
    }
}
