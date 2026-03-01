"""Tests for MH-05: triforce.json multihead model definition and scenario wiring.

Verifies:
- sword-and-beams-multihead model loads from triforce.json
- Model uses MultiHeadAgent neural net and correct action space
- skip-sword-to-triforce scenario has exactly [GainedTriforce, GameOver, Timeout] end conditions
"""

from triforce.models import ModelDefinition, MultiHeadAgent
from triforce.scenario_wrapper import TrainingScenarioDefinition


class TestMultiheadModelDefinition:
    """Verify sword-and-beams-multihead model is correctly defined in triforce.json."""

    def test_model_exists(self):
        """sword-and-beams-multihead should load from triforce.json."""
        model_def = ModelDefinition.get("sword-and-beams-multihead")
        assert model_def is not None, "sword-and-beams-multihead not found in triforce.json"

    def test_model_uses_multihead_agent(self):
        """Model should use MultiHeadAgent as its neural network."""
        model_def = ModelDefinition.get("sword-and-beams-multihead")
        assert model_def.neural_net is MultiHeadAgent

    def test_model_action_space(self):
        """Model should have MOVE, SWORD, BEAMS action space."""
        model_def = ModelDefinition.get("sword-and-beams-multihead")
        assert model_def.action_space == ["MOVE", "SWORD", "BEAMS"]


class TestSkipSwordToTriforceScenario:
    """Verify skip-sword-to-triforce scenario end conditions for multihead training."""

    def test_scenario_exists(self):
        """skip-sword-to-triforce scenario should exist."""
        scenario = TrainingScenarioDefinition.get("skip-sword-to-triforce")
        assert scenario is not None, "skip-sword-to-triforce not found in triforce.json"

    def test_end_conditions_exact(self):
        """End conditions must be exactly [GainedTriforce, GameOver, Timeout].

        No early termination conditions that would cap progress before triforce collection.
        """
        scenario = TrainingScenarioDefinition.get("skip-sword-to-triforce")
        expected = ["GainedTriforce", "GameOver", "Timeout"]
        assert scenario.end_conditions == expected, (
            f"Expected end conditions {expected}, got {scenario.end_conditions}. "
            "Early termination conditions would cap game progress."
        )

    def test_no_early_termination(self):
        """Scenario must not have conditions that stop before dungeon completion."""
        scenario = TrainingScenarioDefinition.get("skip-sword-to-triforce")
        # These conditions would prematurely end episodes before dungeon 1 triforce
        early_termination = [
            "LeftOverworld1Area", "StartingSwordCondition", "EnteredDungeon",
            "LeftDungeon", "NowhereToGoCondition", "LeftPlayArea",
            "Dungeon1DidntGetKey", "LeftWallmasterRoom", "RoomWalkCondition"
        ]
        for condition in early_termination:
            assert condition not in scenario.end_conditions, (
                f"Early termination condition '{condition}' would cap game progress"
            )
