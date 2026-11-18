//! What a heart is worth at every route step (`triforce/heart_value.json`, fitted from chain runs by
//! `python -m triforce.heart_value`), and the reward terms a family's `reward.heart_value` prices hearts with.
//!
//! V(k, h) is the chance that a run entering route step k's room with h hearts is alive when it takes the next
//! triforce, per way of the route (`clips`: the run-time clips switch on; `plain`: off), at every half heart from 0
//! to the step's max hearts; between grid points it is interpolated, beyond the top it is the top's value.  The
//! table names each step's room, entry, exit and visit: a table fitted for another route (the route changed since)
//! is refused by the families that use it ([`HeartValues::check`]) and ignored by everyone else.  A table prices its
//! own route and every route whose steps are its first ones ([`HeartValues::prices`]; round 27: fitted on a scouting
//! route that extends `d2first`, as `d6` was before it merged, it prices `d2first` too): V at a step depends only on
//! the steps up to the next triforce, the same on both routes.

use crate::config::{ConfigError, room_key};
use serde::Deserialize;
use zelda::game_map::{GameMap, RouteStep};

/// One way's curve at a step.
#[derive(Clone, Debug, Deserialize)]
pub struct WayCurve {
    pub max_hearts: u32,
    /// V at 0, 0.5, 1, ... `max_hearts` hearts.
    pub values: Vec<f64>,
    /// Room visits the curve was fitted from.
    #[serde(default)]
    pub visits: u64,
}

impl WayCurve {
    /// V at `hearts`, interpolated between half hearts (clamped to the grid).
    pub fn at(&self, hearts: f64) -> f64 {
        let last = self.values.len() - 1;
        let x = (hearts * 2.0).clamp(0.0, last as f64);
        let i = (x.floor() as usize).min(last);
        let j = (i + 1).min(last);
        let t = x - i as f64;
        self.values[i] * (1.0 - t) + self.values[j] * t
    }

    /// V at the step's max hearts.
    pub fn top(&self) -> f64 {
        self.values[self.values.len() - 1]
    }
}

/// One route step's row.
#[derive(Clone, Debug, Deserialize)]
pub struct StepRow {
    pub step: usize,
    /// `room_key` of the step's room.
    pub room: String,
    pub entry: String,
    pub exit: String,
    pub visit: u32,
    pub clips: Option<WayCurve>,
    pub plain: Option<WayCurve>,
}

/// `triforce/heart_value.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct HeartValues {
    pub route: String,
    pub steps: Vec<StepRow>,
    /// The routes the table prices ([`Self::prices`]), set where the data is loaded (`GameData::load`).
    #[serde(skip)]
    pub routes: Vec<String>,
}

impl HeartValues {
    pub fn parse(text: &str) -> Result<HeartValues, ConfigError> {
        let values: HeartValues =
            serde_json::from_str(text).map_err(|e| ConfigError(format!("triforce/heart_value.json: {e}")))?;
        for (i, row) in values.steps.iter().enumerate() {
            let curves = [&row.clips, &row.plain];
            if row.step != i
                || curves.iter().flat_map(|c| c.as_ref()).any(|c| c.values.len() != 2 * c.max_hearts as usize + 1)
            {
                return Err(ConfigError(format!(
                    "triforce/heart_value.json: step row {i} is malformed (step {}, or a curve whose length is not \
                     2 × max_hearts + 1)",
                    row.step
                )));
            }
        }
        Ok(values)
    }

    /// The table was fitted for `steps`, the route it names as it stands now (same rooms, entries, exits and
    /// visits); refit with `python -m triforce.heart_value` otherwise.
    pub fn check(&self, steps: &[RouteStep], visits: &[u32]) -> Result<(), ConfigError> {
        if self.steps.len() != steps.len() {
            return self.mismatch(format!("{} steps, the route has {}", self.steps.len(), steps.len()));
        }
        self.check_rows(steps, visits)
    }

    /// The table's first rows are `steps` (a route its own extends, or its own route).
    fn check_rows(&self, steps: &[RouteStep], visits: &[u32]) -> Result<(), ConfigError> {
        if steps.len() > self.steps.len() {
            return self.mismatch(format!("{} steps, the route has {}", self.steps.len(), steps.len()));
        }
        for ((row, step), &visit) in self.steps.iter().zip(steps).zip(visits) {
            let room = room_key(step.location());
            if row.room != room || row.entry != step.entry || row.exit != step.exit || row.visit != visit {
                return self.mismatch(format!(
                    "step {}: table {} {}->{} visit {}, route {room} {}->{} visit {visit}",
                    row.step, row.room, row.entry, row.exit, row.visit, step.entry, step.exit
                ));
            }
        }
        Ok(())
    }

    fn mismatch(&self, what: String) -> Result<(), ConfigError> {
        Err(ConfigError(format!(
            "triforce/heart_value.json does not match route {:?} ({what}): refit it with `python -m \
             triforce.heart_value <chain jsonl>...`",
            self.route
        )))
    }

    /// The routes of `map` the table prices: none when it no longer matches its own route ([`Self::check`]), else
    /// its own and every route whose steps are its first rows (one it extends).
    pub fn prices(&self, map: &GameMap) -> Vec<String> {
        let fitted =
            map.routes.get(&self.route).is_some_and(|steps| self.check(steps, &map.visits[&self.route]).is_ok());
        if !fitted {
            return Vec::new();
        }
        map.routes
            .iter()
            .filter(|(name, steps)| self.check_rows(steps, &map.visits[*name]).is_ok())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// The curve of route step `step` on the way the run-time clips switch selects; None off the route or for a step
    /// not on that way.
    pub fn curve(&self, step: Option<usize>, clips: bool) -> Option<&WayCurve> {
        let row = self.steps.get(step?)?;
        if clips { row.clips.as_ref() } else { row.plain.as_ref() }
    }
}

/// Where a scored step was played on the route, and the table that prices hearts there (`reward.heart_value`).
#[derive(Clone, Copy, Debug, Default)]
pub struct HeartContext<'a> {
    pub values: Option<&'a HeartValues>,
    /// The route step of the room the step was played in (before the action).
    pub step: Option<usize>,
    /// The route step after the action (the next room's after an exit).
    pub next: Option<usize>,
    /// The run-time clips switch (which way's curve).
    pub clips: bool,
}

impl HeartContext<'_> {
    /// The exit's health fraction priced by the next step's curve: V(next, hearts) / V(next, its max hearts); None
    /// without a curve there (or one that is 0 at the top).
    pub fn exit_fraction(&self, hearts: f64) -> Option<f64> {
        let curve = self.values?.curve(self.next, self.clips)?;
        let top = curve.top();
        (top > 0.0).then(|| (curve.at(hearts) / top).clamp(0.0, 1.0))
    }

    /// V(room's step, after) − V(room's step, before); None without a curve.
    pub fn change(&self, before: f64, after: f64) -> Option<f64> {
        let curve = self.values?.curve(self.step, self.clips)?;
        Some(curve.at(after) - curve.at(before))
    }
}
