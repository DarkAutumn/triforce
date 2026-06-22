# pylint: disable=all
"""Experiment 4 circuit configuration tests."""

from triforce.scenario_wrapper import TrainingCircuitDefinition


def test_experiment4_weighted_circuit_loads():
    circuit = TrainingCircuitDefinition.get("experiment4-late-chain-demo")

    assert circuit.name == "experiment4-late-chain-demo"
    assert circuit.kind == "weighted"
    assert [entry.scenario for entry in circuit.scenarios] == [
        "dungeon1-wallmaster-north-exit",
        "dungeon1-late-chain",
        "dungeon1-aquamentus-east",
    ]
    assert [entry.weight for entry in circuit.scenarios] == [30, 50, 20]
    assert [entry.exit_criteria.metric for entry in circuit.scenarios] == ["success-rate"] * 3
    assert [entry.exit_criteria.threshold for entry in circuit.scenarios] == [0.5, 0.1, 0.2]


def test_experiment4_sequential_circuit_loads():
    circuit = TrainingCircuitDefinition.get("experiment4-circuit")

    assert circuit.name == "experiment4-circuit"
    assert circuit.final_eval_scenario == "dungeon1-late-chain"
    assert circuit.final_eval_episodes == 100
    assert len(circuit.scenarios) == 2

    late_chain, finite_bombs = circuit.scenarios
    assert late_chain.circuit == "experiment4-late-chain-demo"
    assert late_chain.iterations == 1500000
    assert finite_bombs.scenario == "dungeon1-finite-bombs"
    assert finite_bombs.iterations == 500000
    assert finite_bombs.exit_criteria.metric == "success-rate"
    assert finite_bombs.exit_criteria.threshold == 0.1
