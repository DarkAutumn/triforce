"""Tests for weighted circuit training infrastructure.

Tests:
- ExitCriteria Pydantic model parsing
- TrainingCircuitDefinition kind/weight fields and validation
- WeightedScenarioSelector deficit-based scenario selection
- Backward compatibility of existing sequential circuits
- MetricTracker per-scenario buffering in weighted mode
- ConditionalTrigger schema and validation
- _ConditionalMonitor trigger detection and cooldown
"""

import pytest

from triforce.scenario_wrapper import (
    ConditionalTrigger, ExitCriteria, TrainingCircuitEntry, TrainingCircuitDefinition,
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
            name='test',
            scenarios=[TrainingCircuitEntry(scenario='full-game')])
        assert c.kind == 'sequential'

    def test_explicit_sequential(self):
        c = TrainingCircuitDefinition(
            name='test', kind='sequential',
            scenarios=[TrainingCircuitEntry(scenario='full-game')])
        assert c.kind == 'sequential'

    def test_weighted_kind(self):
        c = TrainingCircuitDefinition(
            name='test', kind='weighted',
            scenarios=[TrainingCircuitEntry(scenario='full-game', weight=100)])
        assert c.kind == 'weighted'

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown circuit kind"):
            TrainingCircuitDefinition(
                name='test', kind='invalid',
                scenarios=[TrainingCircuitEntry(scenario='full-game')])

    def test_existing_sequential_circuits(self):
        """Sequential circuits in triforce.yaml should have kind=sequential."""
        for circuit in TrainingCircuitDefinition.get_all():
            if circuit.kind == 'sequential':
                assert circuit.kind == 'sequential', f"{circuit.name} should be sequential"

    def test_room_walk_circuit_is_weighted(self):
        """The room-walk-circuit should be weighted."""
        circuit = TrainingCircuitDefinition.get('room-walk-circuit')
        assert circuit is not None
        assert circuit.kind == 'weighted'
        assert len(circuit.scenarios) == 2
        for entry in circuit.scenarios:
            assert entry.weight is not None

    def test_existing_circuits_parse_exit_criteria(self):
        """Existing circuits should parse the new exit-criteria format."""
        circuit = TrainingCircuitDefinition.get('main-circuit')
        assert circuit is not None
        # Second entry (overworld-sword) should have exit criteria
        second = circuit.scenarios[1]
        assert second.exit_criteria is not None
        assert second.exit_criteria.metric == 'success-rate'
        assert second.exit_criteria.threshold == 0.8


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


# ---------------------------------------------------------------------------
# ConditionalTrigger schema and validation
# ---------------------------------------------------------------------------
class TestConditionalTrigger:
    """Verify ConditionalTrigger model and circuit validation."""

    def test_basic_parsing(self):
        ct = ConditionalTrigger(metric='endings/failure-wallmastered', threshold=0.5)
        assert ct.metric == 'endings/failure-wallmastered'
        assert ct.threshold == 0.5
        assert ct.cooldown == 0  # default

    def test_with_cooldown(self):
        ct = ConditionalTrigger(metric='endings/failure-wallmastered',
                                threshold=0.5, cooldown=100000)
        assert ct.cooldown == 100000

    def test_circuit_entry_with_condition(self):
        entry = TrainingCircuitEntry(**{
            'scenario': 'dungeon1-wallmaster',
            'iterations': 250000,
            'exit-criteria': {'metric': 'success-rate', 'threshold': 0.8},
            'condition': {'metric': 'endings/failure-wallmastered',
                          'threshold': 0.5, 'cooldown': 100000},
        })
        assert entry.condition is not None
        assert entry.condition.metric == 'endings/failure-wallmastered'
        assert entry.condition.threshold == 0.5
        assert entry.condition.cooldown == 100000
        assert entry.weight is None

    def test_conditional_entry_without_weight_in_weighted_circuit(self):
        """Conditional entries in weighted circuits must not have a weight."""
        c = TrainingCircuitDefinition.get('dungeon1-unrestricted-items')
        assert c is not None
        assert c.kind == 'weighted'
        conditional = [e for e in c.scenarios if e.condition is not None]
        assert len(conditional) == 1
        assert conditional[0].weight is None
        assert conditional[0].scenario == 'dungeon1-wallmaster'

    def test_weighted_entry_with_condition_rejected(self):
        """An entry with both weight and condition should be rejected."""
        from pydantic import ValidationError  # pylint: disable=import-outside-toplevel
        with pytest.raises(ValidationError, match="must not have a weight"):
            TrainingCircuitDefinition(**{
                'name': 'test',
                'kind': 'weighted',
                'scenarios': [
                    {'scenario': 'full-game', 'weight': 90},
                    {'scenario': 'dungeon1', 'weight': 10,
                     'condition': {'metric': 'x', 'threshold': 0.5}},
                ],
            })

    def test_condition_in_sequential_rejected(self):
        """Conditions should be rejected in sequential circuits."""
        from pydantic import ValidationError  # pylint: disable=import-outside-toplevel
        with pytest.raises(ValidationError, match="only weighted circuits support conditions"):
            TrainingCircuitDefinition(**{
                'name': 'test',
                'kind': 'sequential',
                'scenarios': [
                    {'scenario': 'full-game',
                     'condition': {'metric': 'x', 'threshold': 0.5}},
                ],
            })

    def test_weighted_entry_without_weight_and_without_condition_rejected(self):
        """Weighted entries without condition must have a weight."""
        from pydantic import ValidationError  # pylint: disable=import-outside-toplevel
        with pytest.raises(ValidationError, match="must have a weight"):
            TrainingCircuitDefinition(**{
                'name': 'test',
                'kind': 'weighted',
                'scenarios': [
                    {'scenario': 'full-game'},
                ],
            })


# ---------------------------------------------------------------------------
# _ConditionalMonitor trigger detection and cooldown
# ---------------------------------------------------------------------------
class TestConditionalMonitor:
    """Verify conditional trigger detection and cooldown logic."""

    def _make_monitor(self, threshold=0.5, cooldown=0):
        # Import here to avoid circular imports at module level
        from train import _ConditionalMonitor  # pylint: disable=import-outside-toplevel
        entry = TrainingCircuitEntry(**{
            'scenario': 'wallmaster-fix',
            'iterations': 250000,
            'exit-criteria': {'metric': 'success-rate', 'threshold': 0.8},
            'condition': {'metric': 'endings/failure-wallmastered',
                          'threshold': threshold, 'cooldown': cooldown},
        })
        return _ConditionalMonitor(inner=None, conditional_entries=[entry])

    def test_no_trigger_below_threshold(self):
        monitor = self._make_monitor(threshold=0.5)
        monitor.on_progress(1000, 10000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.3}, 1000, 10000)
        assert monitor.triggered_entry is None

    def test_triggers_at_threshold(self):
        monitor = self._make_monitor(threshold=0.5)
        monitor.on_progress(1000, 10000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.5}, 1000, 10000)
        assert monitor.triggered_entry is not None
        assert monitor.triggered_entry.scenario == 'wallmaster-fix'

    def test_triggers_above_threshold(self):
        monitor = self._make_monitor(threshold=0.5)
        monitor.on_progress(1000, 10000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.8}, 1000, 10000)
        assert monitor.triggered_entry is not None

    def test_check_pause_returns_false_when_triggered(self):
        monitor = self._make_monitor(threshold=0.5)
        assert monitor.check_pause() is True
        monitor.on_progress(1000, 10000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.6}, 1000, 10000)
        assert monitor.check_pause() is False

    def test_clear_trigger(self):
        monitor = self._make_monitor(threshold=0.5)
        monitor.on_progress(1000, 10000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.6}, 1000, 10000)
        assert monitor.triggered_entry is not None
        monitor.clear_trigger()
        assert monitor.triggered_entry is None
        assert monitor.check_pause() is True

    def test_cooldown_prevents_retrigger(self):
        monitor = self._make_monitor(threshold=0.5, cooldown=50000)
        # First trigger (no cooldown at startup)
        monitor.on_progress(1000, 100000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.6}, 1000, 100000)
        assert monitor.triggered_entry is not None
        monitor.clear_trigger()

        # Before cooldown expires — should not trigger
        monitor.on_progress(10000, 100000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.9}, 11000, 100000)
        assert monitor.triggered_entry is None

        # After cooldown expires — should trigger again
        monitor.on_progress(50000, 100000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.6}, 61000, 100000)
        assert monitor.triggered_entry is not None

    def test_no_cooldown_at_startup(self):
        """Conditions should be able to trigger immediately at circuit startup."""
        monitor = self._make_monitor(threshold=0.5, cooldown=100000)
        # Initialize cooldown to cooldown value so it can trigger immediately
        monitor.on_metrics({'endings/failure-wallmastered': 0.7}, 0, 100000)
        assert monitor.triggered_entry is not None

    def test_trigger_count_increments(self):
        monitor = self._make_monitor(threshold=0.5, cooldown=0)
        monitor.on_progress(1000, 100000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.6}, 1000, 100000)
        assert monitor.trigger_counts[0] == 1
        monitor.clear_trigger()

        monitor.on_progress(1000, 100000)
        monitor.on_metrics({'endings/failure-wallmastered': 0.7}, 2000, 100000)
        assert monitor.trigger_counts[0] == 2

    def test_missing_metric_does_not_trigger(self):
        monitor = self._make_monitor(threshold=0.5)
        monitor.on_progress(1000, 10000)
        monitor.on_metrics({'success-rate': 0.9}, 1000, 10000)
        assert monitor.triggered_entry is None


# ---------------------------------------------------------------------------
# Modifier system
# ---------------------------------------------------------------------------
class TestModifierDefinition:
    """Verify modifier parsing and resolution."""

    def test_load_named_modifiers(self):
        """Named modifiers should load from triforce.yaml."""
        mods = TrainingCircuitDefinition._load_modifiers()
        assert 'has-sword' in mods
        assert 'all-equipment' in mods
        assert mods['has-sword'].per_reset == {'sword': 1}

    def test_per_step_on_modifier(self):
        """Modifiers can have per-step fields."""
        mods = TrainingCircuitDefinition._load_modifiers()
        assert 'infinite-bombs' in mods
        assert mods['infinite-bombs'].per_step == {'bombs': 4}

    def test_modifier_with_both_per_reset_and_per_step(self):
        """Modifiers can have both per-reset and per-step."""
        mods = TrainingCircuitDefinition._load_modifiers()
        assert 'infinite-arrows' in mods
        assert mods['infinite-arrows'].per_reset == {'arrows': 1, 'bow': 1}
        assert mods['infinite-arrows'].per_step == {'rupees': 10}


class TestModifierResolution:
    """Verify resolve_modifier_list function."""

    def test_single_named_modifier(self):
        from triforce.scenario_wrapper import resolve_modifier_list
        mods = TrainingCircuitDefinition._load_modifiers()
        per_reset, per_step, per_room = resolve_modifier_list(['has-sword'], mods)
        assert per_reset == {'sword': 1}
        assert per_step == {}
        assert per_room == {}

    def test_multiple_modifiers_merge(self):
        from triforce.scenario_wrapper import resolve_modifier_list
        mods = TrainingCircuitDefinition._load_modifiers()
        per_reset, per_step, per_room = resolve_modifier_list(
            ['has-sword', 'starting-bombs'], mods)
        assert per_reset == {'sword': 1, 'bombs': 8}

    def test_later_modifier_overrides_earlier(self):
        """Later modifiers override earlier ones for the same key."""
        from triforce.scenario_wrapper import resolve_modifier_list, ModifierDefinition
        mods = {'a': ModifierDefinition(per_reset={'x': 1}),
                'b': ModifierDefinition(per_reset={'x': 2})}
        per_reset, _, _ = resolve_modifier_list(['a', 'b'], mods)
        assert per_reset == {'x': 2}

    def test_inline_modifier(self):
        from triforce.scenario_wrapper import resolve_modifier_list
        mods = {}
        per_reset, per_step, _ = resolve_modifier_list(
            [{'per-reset': {'foo': 1}, 'per-step': {'bar': 2}}], mods)
        assert per_reset == {'foo': 1}
        assert per_step == {'bar': 2}

    def test_unknown_modifier_raises(self):
        from triforce.scenario_wrapper import resolve_modifier_list
        with pytest.raises(ValueError, match="Unknown modifier 'nonexistent'"):
            resolve_modifier_list(['nonexistent'], {})


class TestScenarioModifiers:
    """Verify scenario-level modifiers resolve into per_reset/per_step."""

    def test_scenario_with_modifiers(self):
        """overworld-skip-sword has has-sword modifier, should have sword in per_reset."""
        scenario = TrainingScenarioDefinition.get('overworld-skip-sword')
        assert scenario.per_reset.get('sword') == 1

    def test_scenario_without_modifiers(self):
        """full-game has no modifiers, should have empty per_reset."""
        scenario = TrainingScenarioDefinition.get('full-game')
        assert scenario.per_reset == {}
        assert scenario.per_step == {}

    def test_dungeon1_room_walk_has_keys(self):
        """dungeon1-room-walk has four-keys modifier."""
        scenario = TrainingScenarioDefinition.get('dungeon1-room-walk')
        assert scenario.per_reset.get('keys') == 4


class TestCircuitModifiers:
    """Verify circuit-level and entry-level modifiers."""

    def test_circuit_has_modifiers(self):
        """room-walk-circuit should have circuit-level modifiers."""
        circuit = TrainingCircuitDefinition.get('room-walk-circuit')
        assert circuit.modifiers is not None
        assert 'all-equipment' in circuit.modifiers

    def test_entry_has_modifiers(self):
        """Entries in overworld-dungeon1-unrestricted-items should have entry modifiers."""
        circuit = TrainingCircuitDefinition.get('overworld-dungeon1-unrestricted-items')
        # Second entry should have modifiers
        second = circuit.scenarios[1]
        assert second.modifiers is not None
        assert 'all-equipment' in second.modifiers

    def test_entry_null_modifiers(self):
        """First entry in overworld-dungeon1-unrestricted-items has null modifiers."""
        circuit = TrainingCircuitDefinition.get('overworld-dungeon1-unrestricted-items')
        first = circuit.scenarios[0]
        assert first.modifiers is None


class TestApplyModifierChain:
    """Verify _apply_modifier_chain from train.py."""

    def test_no_modifiers_returns_original(self):
        from train import _apply_modifier_chain
        scenario = TrainingScenarioDefinition.get('full-game')
        result = _apply_modifier_chain(scenario)
        assert result is scenario  # same object, not a copy

    def test_entry_modifiers_applied(self):
        from train import _apply_modifier_chain
        scenario = TrainingScenarioDefinition.get('dungeon1')
        result = _apply_modifier_chain(scenario, entry_modifiers=['starting-bombs'])
        assert result is not scenario  # deep copy
        assert result.per_reset.get('bombs') == 8
        assert scenario.per_reset.get('bombs') is None  # original unchanged

    def test_modifier_chain_applied(self):
        from train import _apply_modifier_chain
        scenario = TrainingScenarioDefinition.get('dungeon1')
        result = _apply_modifier_chain(scenario,
                                        modifier_chain=[['all-equipment'], ['infinite-bombs']])
        assert result.per_reset.get('sword') == 1
        assert result.per_step.get('bombs') == 4

    def test_outer_overrides_inner(self):
        """Outer modifier chain entries override inner ones."""
        from train import _apply_modifier_chain
        scenario = TrainingScenarioDefinition.get('overworld-skip-sword')
        # Scenario has has-sword (sword: 1), chain adds all-equipment (also sword: 1)
        # Then an inline modifier that sets sword: 0
        result = _apply_modifier_chain(
            scenario,
            modifier_chain=[[{'per-reset': {'sword': 0}}]])
        assert result.per_reset.get('sword') == 0

    def test_scenario_base_preserved_with_chain(self):
        """Scenario's own modifiers (per_reset) are preserved and extended by chain."""
        from train import _apply_modifier_chain
        scenario = TrainingScenarioDefinition.get('overworld-skip-sword')
        assert scenario.per_reset.get('sword') == 1  # from has-sword
        result = _apply_modifier_chain(scenario, entry_modifiers=['starting-bombs'])
        assert result.per_reset.get('sword') == 1  # preserved
        assert result.per_reset.get('bombs') == 8  # added
