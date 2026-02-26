"""Tests for the FutureCreditLedger discount system.

Verifies that:
- Predictions are recorded and consumed correctly
- Multiple independent predictions don't merge
- Stale predictions expire after PREDICTION_EXPIRY_FRAMES
- Room transitions clear the ledger
- Edge cases: double-kill, missed predictions, overlapping weapons
"""

import pytest
from triforce.state_change_wrapper import FutureCreditLedger, Prediction, PREDICTION_EXPIRY_FRAMES
from triforce.zelda_enums import ZeldaItemKind


class TestPrediction:
    """Test the Prediction dataclass."""

    def test_empty_prediction(self):
        p = Prediction(frame=0)
        assert p.is_empty

    def test_nonempty_with_hits(self):
        p = Prediction(frame=0, enemy_hits={1: 3})
        assert not p.is_empty

    def test_nonempty_with_stuns(self):
        p = Prediction(frame=0, enemy_stuns=[2])
        assert not p.is_empty

    def test_nonempty_with_items(self):
        p = Prediction(frame=0, items=[ZeldaItemKind.Heart])
        assert not p.is_empty

    def test_becomes_empty_after_consuming_hits(self):
        p = Prediction(frame=0, enemy_hits={1: 3})
        del p.enemy_hits[1]
        assert p.is_empty


class TestFutureCreditLedger:
    """Test the FutureCreditLedger discount system."""

    def test_no_predictions_no_discount(self):
        """With no predictions, actuals should be unchanged."""
        ledger = FutureCreditLedger()
        hits = {1: 3}
        stunned = [2]
        items = [ZeldaItemKind.Heart]
        ledger.discount(100, hits, stunned, items)
        assert hits == {1: 3}
        assert stunned == [2]
        assert items == [ZeldaItemKind.Heart]

    def test_basic_discount(self):
        """A prediction should discount matching actual damage."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 3}))

        hits = {1: 3}
        ledger.discount(20, hits, [], [])
        assert hits == {}  # fully consumed

    def test_partial_discount(self):
        """Prediction for 3 damage, actual is 5 → 2 remaining."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 3}))

        hits = {1: 5}
        ledger.discount(20, hits, [], [])
        assert hits == {1: 2}

    def test_prediction_consumed_after_discount(self):
        """After full discount, prediction should be gone — second actual gets no discount."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 3}))

        # First actual: consumes the prediction
        hits1 = {1: 3}
        ledger.discount(20, hits1, [], [])
        assert hits1 == {}

        # Second actual: no prediction left to discount
        hits2 = {1: 3}
        ledger.discount(30, hits2, [], [])
        assert hits2 == {1: 3}

    def test_independent_predictions_dont_merge(self):
        """Two predictions for the same enemy should be independent."""
        ledger = FutureCreditLedger()
        # Beam predicts 3 damage to enemy 1
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 3}))
        # Bomb also predicts 2 damage to enemy 1
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 2}))

        # Actual: enemy 1 takes 3 damage (from beam)
        hits = {1: 3}
        ledger.discount(20, hits, [], [])
        # First prediction fully consumed (3-3=0), second untouched → net 0
        assert hits == {}

        # Later: enemy 1 takes 2 more damage (from bomb)
        hits2 = {1: 2}
        ledger.discount(30, hits2, [], [])
        # Second prediction consumed
        assert hits2 == {}

        # No more predictions — third actual is not discounted
        hits3 = {1: 1}
        ledger.discount(40, hits3, [], [])
        assert hits3 == {1: 1}

    def test_missed_prediction_expires(self):
        """A prediction that never materializes should expire after PREDICTION_EXPIRY_FRAMES."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 3}))

        # No actual damage for a long time...
        hits = {}
        ledger.discount(10 + PREDICTION_EXPIRY_FRAMES, hits, [], [])
        # Prediction expired

        # Real damage later should NOT be discounted
        hits2 = {1: 3}
        ledger.discount(10 + PREDICTION_EXPIRY_FRAMES + 1, hits2, [], [])
        assert hits2 == {1: 3}

    def test_clear_removes_all_predictions(self):
        """clear() should remove all predictions (e.g., on room change)."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 3}))
        ledger.add_prediction(Prediction(frame=10, enemy_hits={2: 5}))

        ledger.clear()

        hits = {1: 3, 2: 5}
        ledger.discount(20, hits, [], [])
        assert hits == {1: 3, 2: 5}  # no discount

    def test_stun_discount(self):
        """Predicted stuns should be discounted."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10, enemy_stuns=[1, 2]))

        stunned = [1, 2]
        ledger.discount(20, {}, stunned, [])
        assert stunned == []

    def test_item_discount(self):
        """Predicted item pickups should be discounted."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10, items=[ZeldaItemKind.Heart]))

        items = [ZeldaItemKind.Heart]
        ledger.discount(20, {}, [], items)
        assert items == []

    def test_empty_prediction_not_added(self):
        """Adding an empty prediction should be a no-op."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10))
        assert len(ledger._predictions) == 0  # pylint: disable=protected-access

    def test_different_enemies_not_cross_discounted(self):
        """Prediction for enemy 1 should not discount damage to enemy 2."""
        ledger = FutureCreditLedger()
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 3}))

        hits = {2: 3}
        ledger.discount(20, hits, [], [])
        assert hits == {2: 3}  # no discount — wrong enemy

    def test_bomb_then_sword_then_second_bomb(self):
        """User's edge case: bomb → sword kills → second bomb.

        1. Bomb drop → predict kill enemy A (credit bomb step)
        2. Sword kills enemy A → ledger discounts it (net 0)
        3. Second bomb → predict kill enemy B (credit second bomb step)
        4. Enemy B dies → ledger discounts it (net 0)
        No cross-contamination between predictions.
        """
        ledger = FutureCreditLedger()

        # Step 1: Bomb predicts killing enemy A (index 1, 3 HP)
        ledger.add_prediction(Prediction(frame=10, enemy_hits={1: 3}))

        # Step 2: Sword actually kills enemy A
        hits = {1: 3}
        ledger.discount(15, hits, [], [])
        assert hits == {}  # discounted — already credited to bomb

        # Step 3: Second bomb predicts killing enemy B (index 2, 2 HP)
        ledger.add_prediction(Prediction(frame=20, enemy_hits={2: 2}))

        # Step 4: Enemy B dies
        hits2 = {2: 2}
        ledger.discount(30, hits2, [], [])
        assert hits2 == {}  # discounted — already credited to second bomb

        # No stale discounts left
        hits3 = {3: 1}
        ledger.discount(40, hits3, [], [])
        assert hits3 == {3: 1}  # not discounted
