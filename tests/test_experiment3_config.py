# pylint: disable=all
"""Experiment 3 circuit configuration tests."""

from triforce.scenario_wrapper import TrainingCircuitDefinition


def test_experiment3_circuit_loads():
    circuit = TrainingCircuitDefinition.get("experiment3-circuit")

    assert circuit.name == "experiment3-circuit"
    assert circuit.final_eval_scenario == "dungeon1-wallmaster-north-exit"
    assert circuit.final_eval_episodes == 100


def test_experiment3_circuit_entries():
    circuit = TrainingCircuitDefinition.get("experiment3-circuit")

    assert len(circuit.scenarios) == 1
    entry = circuit.scenarios[0]
    assert entry.scenario == "dungeon1-wallmaster-north-exit"
    assert entry.iterations == 750000
    assert entry.exit_criteria.metric == "success-rate"
    assert entry.exit_criteria.threshold == 0.5
