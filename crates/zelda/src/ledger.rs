//! Look-ahead predictions and their discounting (`Prediction`, `FutureCreditLedger`).

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
    /// The weapon whose look-ahead made it: its unconsumed entries go when that weapon is seen inactive again.
    pub weapon: Option<crate::enums::AnimationKind>,
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

    /// Drops what weapons that have finished still had pending (`finished(weapon)`): such a prediction named a hit
    /// that didn't land as predicted (another slot was hit, or nothing), and must not swallow a later real hit.
    pub fn drop_finished(&mut self, finished: impl Fn(crate::enums::AnimationKind) -> bool) {
        self.predictions.retain(|p| p.weapon.is_none_or(|w| !finished(w)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prediction(frame: u64, hits: &[(u8, i32)], stuns: &[u8], items: &[i16]) -> Prediction {
        Prediction {
            frame,
            enemy_hits: hits.to_vec(),
            enemy_stuns: stuns.to_vec(),
            items: items.to_vec(),
            weapon: None,
        }
    }

    /// (hits, stuns, items) left after discounting the observations at `frame`.
    fn discount(
        ledger: &mut FutureCreditLedger,
        frame: u64,
        hits: &[(u8, i32)],
        stuns: &[u8],
        items: &[i16],
    ) -> (Vec<(u8, i32)>, Vec<u8>, Vec<i16>) {
        let (mut h, mut s, mut i) = (hits.to_vec(), stuns.to_vec(), items.to_vec());
        ledger.discount(frame, &mut h, &mut s, &mut i);
        (h, s, i)
    }

    #[test]
    fn add_hit_accumulates_per_slot_in_insertion_order() {
        let mut hits = Vec::new();
        add_hit(&mut hits, 3, 1);
        add_hit(&mut hits, 1, 2);
        add_hit(&mut hits, 3, 4);
        assert_eq!(hits, vec![(3, 5), (1, 2)]);
    }

    #[test]
    fn empty_predictions_are_not_recorded() {
        let mut ledger = FutureCreditLedger::default();
        ledger.add_prediction(prediction(10, &[], &[], &[]));
        assert!(ledger.predictions().is_empty());
        ledger.add_prediction(prediction(10, &[], &[2], &[]));
        assert_eq!(ledger.predictions().len(), 1);
    }

    #[test]
    fn predicted_effects_are_not_credited_twice() {
        let mut ledger = FutureCreditLedger::default();
        ledger.add_prediction(prediction(100, &[(2, 3)], &[4], &[0x18]));
        // The beam lands later: the observed hit, stun and pickup were credited at the firing step.
        let left = discount(&mut ledger, 120, &[(2, 3), (5, 1)], &[4, 6], &[0x18, 0x22]);
        assert_eq!(left, (vec![(5, 1)], vec![6], vec![0x22]));
        assert!(
            ledger.predictions().is_empty(),
            "a fully observed prediction is consumed"
        );
        // Once consumed, the same effects are new again.
        let left = discount(&mut ledger, 130, &[(2, 3)], &[4], &[0x18]);
        assert_eq!(left, (vec![(2, 3)], vec![4], vec![0x18]));
    }

    #[test]
    fn partial_hits_carry_the_remainder_either_way() {
        let mut ledger = FutureCreditLedger::default();
        ledger.add_prediction(prediction(0, &[(2, 3)], &[], &[]));
        // Observed less than predicted: nothing new, 2 damage still predicted.
        assert_eq!(discount(&mut ledger, 5, &[(2, 1)], &[], &[]).0, vec![]);
        assert_eq!(ledger.predictions()[0].enemy_hits, vec![(2, 2)]);
        // Observed more than what remains: only the excess is new.
        assert_eq!(
            discount(&mut ledger, 6, &[(2, 5)], &[], &[]).0,
            vec![(2, 3)]
        );
        assert!(ledger.predictions().is_empty());
    }

    #[test]
    fn duplicate_ids_are_matched_one_for_one() {
        let mut ledger = FutureCreditLedger::default();
        ledger.add_prediction(prediction(0, &[], &[3], &[0x18]));
        let left = discount(&mut ledger, 1, &[], &[3, 3], &[0x18, 0x18]);
        assert_eq!((left.1, left.2), (vec![3], vec![0x18]));
    }

    #[test]
    fn each_prediction_discounts_its_own_share() {
        let mut ledger = FutureCreditLedger::default();
        ledger.add_prediction(prediction(0, &[(1, 1)], &[], &[]));
        ledger.add_prediction(prediction(4, &[(1, 2)], &[], &[]));
        assert_eq!(
            discount(&mut ledger, 8, &[(1, 4)], &[], &[]).0,
            vec![(1, 1)]
        );
        assert!(ledger.predictions().is_empty());
    }

    #[test]
    fn predictions_expire_after_200_frames() {
        let mut ledger = FutureCreditLedger::default();
        ledger.add_prediction(prediction(1000, &[(2, 1)], &[], &[]));
        // 199 frames later the prediction still applies.
        assert_eq!(discount(&mut ledger, 1199, &[(2, 1)], &[], &[]).0, vec![]);

        ledger.add_prediction(prediction(1000, &[(2, 1)], &[], &[]));
        // 200 frames later it is dropped and the hit counts.
        assert_eq!(
            discount(&mut ledger, 1200, &[(2, 1)], &[], &[]).0,
            vec![(2, 1)]
        );
        assert!(ledger.predictions().is_empty());
    }

    #[test]
    fn unmatched_predictions_wait_and_clear_drops_them() {
        let mut ledger = FutureCreditLedger::default();
        ledger.add_prediction(prediction(0, &[(2, 1)], &[], &[]));
        assert_eq!(
            discount(&mut ledger, 10, &[(3, 1)], &[], &[]).0,
            vec![(3, 1)]
        );
        assert_eq!(ledger.predictions().len(), 1);
        ledger.clear();
        assert_eq!(
            discount(&mut ledger, 11, &[(2, 1)], &[], &[]).0,
            vec![(2, 1)]
        );
    }
}
