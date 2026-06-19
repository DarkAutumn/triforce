# pylint: disable=all
"""Experiment 2 scenario and circuit configuration tests."""

from triforce.scenario_wrapper import TrainingCircuitDefinition


def test_experiment2_circuit_loads():
    circuit = TrainingCircuitDefinition.get("experiment2-circuit")

    assert circuit.name == "experiment2-circuit"
    assert circuit.final_eval_scenario == "dungeon1-wallmaster-north-exit"
    assert circuit.final_eval_episodes == 100


def test_experiment2_circuit_entries():
    circuit = TrainingCircuitDefinition.get("experiment2-circuit")

    assert [(entry.scenario, entry.iterations) for entry in circuit.scenarios] == [
        ("dungeon1-room-walk", 500000),
        ("dungeon1-wallmaster-north-exit", 750000),
    ]
    assert circuit.scenarios[0].exit_criteria.metric == "room-result/correct-exit"
    assert circuit.scenarios[0].exit_criteria.threshold == 0.8
    assert circuit.scenarios[1].exit_criteria.metric == "success-rate"
    assert circuit.scenarios[1].exit_criteria.threshold == 0.5
