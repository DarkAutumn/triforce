# pylint: disable=all
"""Experiment 1 scenario and circuit configuration tests."""

from triforce.scenario_wrapper import TrainingCircuitDefinition, TrainingScenarioDefinition


def test_experiment1_scenarios_load():
    expected = {
        "dungeon1-red-goriya-east",
        "dungeon1-wallmaster-north-exit",
        "dungeon1-aquamentus-east",
        "dungeon1-late-chain",
    }

    for name in expected:
        assert TrainingScenarioDefinition.get(name) is not None


def test_experiment1_starts_are_exact():
    assert TrainingScenarioDefinition.get("dungeon1-red-goriya-east").start == ["1_44w"]
    assert TrainingScenarioDefinition.get("dungeon1-wallmaster-north-exit").start == ["1_45w"]
    assert TrainingScenarioDefinition.get("dungeon1-aquamentus-east").start == ["1_35s"]
    assert TrainingScenarioDefinition.get("dungeon1-late-chain").start == ["1_43e", "1_44w", "1_45w", "1_35s"]

    for name in (
            "dungeon1-red-goriya-east", "dungeon1-wallmaster-north-exit",
            "dungeon1-aquamentus-east", "dungeon1-late-chain"):
        starts = TrainingScenarioDefinition.get(name).start
        assert not any(start.endswith("c") for start in starts)
        assert not any(start.startswith("debug_") for start in starts)


def test_experiment1_weighted_circuit():
    circuit = TrainingCircuitDefinition.get("dungeon1-endgame-skills")

    assert circuit.kind == "weighted"
    assert [(entry.scenario, entry.weight) for entry in circuit.scenarios] == [
        ("dungeon1-red-goriya-east", 25),
        ("dungeon1-wallmaster-north-exit", 35),
        ("dungeon1-aquamentus-east", 25),
        ("dungeon1-late-chain", 15),
    ]
    assert [entry.exit_criteria.threshold for entry in circuit.scenarios] == [0.7, 0.7, 0.5, 0.2]


def test_experiment1_circuit_final_eval_metadata():
    circuit = TrainingCircuitDefinition.get("experiment1-circuit")

    assert circuit.final_eval_scenario == "full-game-all-items-finite"
    assert circuit.final_eval_episodes == 100
    assert [(entry.circuit, entry.scenario, entry.iterations) for entry in circuit.scenarios] == [
        ("dungeon1-endgame-skills", None, 2500000),
        (None, "dungeon1-finite-bombs", 1000000),
        (None, "full-game-all-items-finite", 1000000),
    ]
