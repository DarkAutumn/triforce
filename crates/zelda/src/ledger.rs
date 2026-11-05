//! Look-ahead predictions and their discounting (`state_change.Prediction`, `FutureCreditLedger`).

/// Predictions older than this many frames are discarded.
pub const PREDICTION_EXPIRY_FRAMES: u64 = 200;

/// Effects one look-ahead predicted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Prediction {
    pub frame: u64,
    /// (slot, damage) in insertion order.
    pub enemy_hits: Vec<(u8, i32)>,
    pub enemy_stuns: Vec<u8>,
    pub items: Vec<i16>,
}

impl Prediction {
    pub fn is_empty(&self) -> bool {
        self.enemy_hits.is_empty() && self.enemy_stuns.is_empty() && self.items.is_empty()
    }
}

/// Adds `dmg` to `slot` of an insertion-ordered (slot, damage) map.
pub fn add_hit(hits: &mut Vec<(u8, i32)>, slot: u8, dmg: i32) {
    match hits.iter_mut().find(|(s, _)| *s == slot) {
        Some((_, d)) => *d += dmg,
        None => hits.push((slot, dmg)),
    }
}

fn remove_first<T: PartialEq>(v: &mut Vec<T>, x: &T) -> bool {
    match v.iter().position(|y| y == x) {
        Some(i) => {
            v.remove(i);
            true
        }
        None => false,
    }
}

/// Predicted effects not yet observed, so they aren't rewarded twice.
#[derive(Clone, Debug, Default)]
pub struct FutureCreditLedger {
    predictions: Vec<Prediction>,
}

impl FutureCreditLedger {
    pub fn clear(&mut self) {
        self.predictions.clear();
    }

    pub fn predictions(&self) -> &[Prediction] {
        &self.predictions
    }

    pub fn add_prediction(&mut self, prediction: Prediction) {
        if !prediction.is_empty() {
            self.predictions.push(prediction);
        }
    }

    /// Subtracts already-predicted effects from this step's observations (in place); drops expired and consumed
    /// predictions.
    pub fn discount(
        &mut self,
        current_frame: u64,
        enemies_hit: &mut Vec<(u8, i32)>,
        enemies_stunned: &mut Vec<u8>,
        items_gained: &mut Vec<i16>,
    ) {
        self.predictions
            .retain(|p| current_frame.saturating_sub(p.frame) < PREDICTION_EXPIRY_FRAMES);
        for pred in &mut self.predictions {
            let slots: Vec<u8> = pred.enemy_hits.iter().map(|(s, _)| *s).collect();
            for slot in slots {
                let Some(ai) = enemies_hit.iter().position(|(s, _)| *s == slot) else {
                    continue;
                };
                let pi = pred
                    .enemy_hits
                    .iter()
                    .position(|(s, _)| *s == slot)
                    .unwrap();
                let subtract = enemies_hit[ai].1.min(pred.enemy_hits[pi].1);
                enemies_hit[ai].1 -= subtract;
                pred.enemy_hits[pi].1 -= subtract;
                if enemies_hit[ai].1 <= 0 {
                    enemies_hit.remove(ai);
                }
                if pred.enemy_hits[pi].1 <= 0 {
                    pred.enemy_hits.remove(pi);
                }
            }
            for slot in pred.enemy_stuns.clone() {
                if remove_first(enemies_stunned, &slot) {
                    remove_first(&mut pred.enemy_stuns, &slot);
                }
            }
            for item in pred.items.clone() {
                if remove_first(items_gained, &item) {
                    remove_first(&mut pred.items, &item);
                }
            }
        }
        self.predictions.retain(|p| !p.is_empty());
    }
}
