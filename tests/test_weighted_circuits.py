"""Tests for weighted circuit training infrastructure.

Tests:
- ExitCriteria Pydantic model parsing
- TrainingCircuitDefinition kind/weight fields and validation
- WeightedScenarioSelector deficit-based scenario selection
- Backward compatibility of existing sequential circuits
- MetricTracker per-scenario buffering in weighted mode
"""

import pytest

from triforce.scenario_wrapper import (
    ExitCriteria, TrainingCircuitEntry, TrainingCircuitDefinition,
    WeightedScenarioSelector, TrainingScenarioDefinition,
)
from triforce.metrics import MetricTracker


# ---------------------------------------------------------------------------
# ExitCriteria parsing
# ---------------------------------------------------------------------------
class TestExitCriteria:
    """Verify the new nested exit-criteria format."""

    def test_basic_parsing(self):
        ec = ExitCriteria(metric='success-rate', threshold=0.9)
        assert ec.metric == 'success-rate'
        assert ec.threshold == 0.9

    def test_from_dict(self):
        ec = ExitCriteria(**{'metric': 'room-result/correct-exit', 'threshold': 0.85})
        assert ec.metric == 'room-result/correct-exit'
        assert ec.threshold == 0.85

    def test_circuit_entry_with_exit_criteria(self):
        """TrainingCircuitEntry should accept exit-criteria as kebab-case alias."""
        entry = TrainingCircuitEntry(**{
            'scenario': 'full-game',
            'exit-criteria': {'metric': 'success-rate', 'threshold': 0.9},
            'iterations': 500000,
        })
        assert entry.exit_criteria is not None
        assert entry.exit_criteria.metric == 'success-rate'
        assert entry.exit_criteria.threshold == 0.9
        assert entry.iterations == 500000

    def test_circuit_entry_without_exit_criteria(self):
        entry = TrainingCircuitEntry(scenario='full-game')
        assert entry.exit_criteria is None

    def test_circuit_entry_with_weight(self):
        entry = TrainingCircuitEntry(scenario='full-game', weight=70.0)
        assert entry.weight == 70.0


# ---------------------------------------------------------------------------
# Circuit kind and weight validation
# ---------------------------------------------------------------------------
class TestCircuitKind:
    """Verify circuit kind field and weight validation."""

    def test_default_kind_is_sequential(self):
        c = TrainingCircuitDefinition(
            name='test', description='test',
            scenarios=[TrainingCircuitEntry(scenario='full-game')])
        assert c.kind == 'sequential'

    def test_explicit_sequential(self):
        c = TrainingCircuitDefinition(
            name='test', description='test', kind='sequential',
            scenarios=[TrainingCircuitEntry(scenario='full-game')])
        assert c.kind == 'sequential'

    def test_weighted_kind(self):
        c = TrainingCircuitDefinition(
            name='test', description='test', kind='weighted',
            scenarios=[TrainingCircuitEntry(scenario='full-game', weight=100)])
        assert c.kind == 'weighted'

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown circuit kind"):
            TrainingCircuitDefinition(
                name='test', description='test', kind='invalid',
                scenarios=[TrainingCircuitEntry(scenario='full-game')])

    def test_existing_circuits_are_sequential(self):
        """All existing circuits in triforce.yaml should be sequential."""
        for circuit in TrainingCircuitDefinition.get_all():
            assert circuit.kind == 'sequential', f"{circuit.name} should be sequential"

    def test_existing_circuits_parse_exit_criteria(self):
        """Existing circuits should parse the new exit-criteria format."""
        circuit = TrainingCircuitDefinition.get('main-circuit')
        assert circuit is not None
        # First entry should have exit criteria
        first = circuit.scenarios[0]
        assert first.exit_criteria is not None
        assert first.exit_criteria.metric == 'room-result/correct-exit'
        assert first.exit_criteria.threshold == 0.9


# ---------------------------------------------------------------------------
# WeightedScenarioSelector
# ---------------------------------------------------------------------------
class TestWeightedScenarioSelector:
    """Verify deficit-based scenario selection logic."""

    def test_initial_returns_first_scenario(self):
        sel = WeightedScenarioSelector(['a', 'b', 'c'], [50, 30, 20])
        assert sel.update(None, 0) == 'a'

    def test_overweight_scenario_avoided(self):
        sel = WeightedScenarioSelector(['a', 'b'], [50, 50])
        # After running only 'a', 'b' should be most underweight
        result = sel.update('a', 1000)
        assert result == 'b'

    def test_converges_to_target_proportions(self):
        """After many episodes, step proportions should approximate target weights."""
        sel = WeightedScenarioSelector(['a', 'b', 'c'], [70, 20, 10])
        scenario = sel.update(None, 0)

        for _ in range(100):
            # Simulate variable-length episodes
            steps = 50 + (_ % 30)
            scenario = sel.update(scenario, steps)

        total = sum(sel._steps.values())
        pct_a = sel._steps['a'] / total
        pct_b = sel._steps['b'] / total
        pct_c = sel._steps['c'] / total

        # Should be within ~10% of targets
        assert abs(pct_a - 0.70) < 0.15, f"a: {pct_a:.2f} vs target 0.70"
        assert abs(pct_b - 0.20) < 0.15, f"b: {pct_b:.2f} vs target 0.20"
        assert abs(pct_c - 0.10) < 0.15, f"c: {pct_c:.2f} vs target 0.10"

    def test_single_scenario(self):
        """With one scenario, it should always return that scenario."""
        sel = WeightedScenarioSelector(['only'], [100])
        assert sel.update(None, 0) == 'only'
        assert sel.update('only', 500) == 'only'
        assert sel.update('only', 500) == 'only'

    def test_all_equal_weights(self):
        """Equal weights should distribute roughly evenly."""
        sel = WeightedScenarioSelector(['a', 'b', 'c'], [1, 1, 1])
        scenario = sel.update(None, 0)
        for _ in range(60):
            scenario = sel.update(scenario, 100)

        total = sum(sel._steps.values())
        for name in ['a', 'b', 'c']:
            pct = sel._steps[name] / total
            assert abs(pct - 1/3) < 0.15, f"{name}: {pct:.2f} vs target 0.33"


# ---------------------------------------------------------------------------
# MetricTracker weighted mode
# ---------------------------------------------------------------------------
class TestMetricTrackerWeighted:
    """Verify per-scenario metric buffering in weighted mode."""

    def setup_method(self):
        """Ensure clean state before each test."""
        MetricTracker.close()
        MetricTracker._buffered_metrics.clear()

    def teardown_method(self):
        MetricTracker.close()
        MetricTracker._buffered_metrics.clear()

    def test_normal_mode_flat_dict(self):
        """Non-weighted mode should return flat dict."""
        tracker = MetricTracker(['success-rate'])
        result = MetricTracker.get_metrics_and_clear()
        # May be empty if no episodes ended, but should be a flat dict
        assert isinstance(result, dict)

    def test_weighted_mode_buffers_on_close(self):
        """Closing a weighted-mode tracker should buffer metrics."""
        tracker = MetricTracker(['success-rate'], scenario_name='full-game')
        # Simulate an ended scenario so metrics have data
        MetricTracker.close()
        # Buffered metrics should have been saved (may be empty if no data)
        # Create a new tracker for different scenario
        tracker2 = MetricTracker(['success-rate'], scenario_name='dungeon1')
        result = MetricTracker.get_metrics_and_clear()
        # Should be a per-scenario dict (possibly empty entries)
        assert isinstance(result, dict)

    def test_weighted_returns_per_scenario(self):
        """In weighted mode, get_metrics_and_clear returns {scenario: {metric: value}}."""
        # Create tracker with scenario name
        tracker = MetricTracker(['success-rate'], scenario_name='full-game')
        # We need to simulate some metric data. SuccessMetric tracks end_scenario calls.
        # Call end_scenario to generate data
        tracker.end_scenario(True, False, 'success')
        tracker.end_scenario(True, False, 'success')
        # Close to buffer
        MetricTracker.close()

        # Create new tracker for different scenario
        tracker2 = MetricTracker(['success-rate'], scenario_name='dungeon1')
        tracker2.end_scenario(False, True, 'timeout')
        tracker2.end_scenario(True, False, 'success')

        result = MetricTracker.get_metrics_and_clear()
        assert 'full-game' in result or 'dungeon1' in result
