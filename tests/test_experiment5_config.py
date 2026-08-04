# pylint: disable=all
"""Experiment 5 circuit configuration tests."""

from triforce.scenario_wrapper import TrainingCircuitDefinition


def test_experiment5_weighted_circuit_loads():
    circuit = TrainingCircuitDefinition.get("experiment5-boss-transfer")

    assert circuit.name == "experiment5-boss-transfer"
    assert circuit.kind == "weighted"
    assert [entry.scenario for entry in circuit.scenarios] == [
        "dungeon1-wallmaster-north-exit",
        "dungeon1-aquamentus-east",
        "dungeon1-late-chain",
    ]
    assert [entry.weight for entry in circuit.scenarios] == [10, 80, 10]
    assert [entry.primary for entry in circuit.scenarios] == [False, True, False]
    assert [entry.exit_criteria.metric for entry in circuit.scenarios] == ["success-rate"] * 3
    assert [entry.exit_criteria.threshold for entry in circuit.scenarios] == [0.8, 0.6, 0.1]


def test_experiment5_sequential_circuit_loads():
    circuit = TrainingCircuitDefinition.get("experiment5-circuit")

    assert circuit.name == "experiment5-circuit"
    assert circuit.final_eval_scenario == "dungeon1-late-chain"
    assert circuit.final_eval_episodes == 100
    assert len(circuit.scenarios) == 2

    boss_transfer, late_chain = circuit.scenarios
    assert boss_transfer.circuit == "experiment5-boss-transfer"
    assert boss_transfer.iterations == 2000000
    assert late_chain.scenario == "dungeon1-late-chain"
    assert late_chain.iterations == 1000000
    assert late_chain.exit_criteria.metric == "success-rate"
    assert late_chain.exit_criteria.threshold == 0.1
