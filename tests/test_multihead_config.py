"""Tests for triforce.yaml config: action spaces, model kinds, and scenario wiring.

Verifies:
- Action space definitions load correctly from triforce.yaml
- Model kind definitions load correctly and map to the right classes
- Defaults are set correctly
- skip-sword-to-triforce scenario has exactly [GainedTriforce, GameOver, Timeout] end conditions
"""

from triforce.models import ActionSpaceDefinition, ModelKindDefinition, MultiHeadAgent, SharedNatureAgent
from triforce.scenario_wrapper import TrainingScenarioDefinition


class TestActionSpaceDefinitions:
    """Verify action-spaces section of triforce.yaml."""

    def test_basic_action_space(self):
        """basic action space should have MOVE, SWORD, BEAMS."""
        asd = ActionSpaceDefinition.get("basic")
        assert asd.actions == ["MOVE", "SWORD", "BEAMS"]

    def test_basic_is_default(self):
        """basic should be the default action space."""
        asd = ActionSpaceDefinition.get_default()
        assert asd.name == "basic"

    def test_move_only_action_space(self):
        """move-only action space should have just MOVE."""
        asd = ActionSpaceDefinition.get("move-only")
        assert asd.actions == ["MOVE"]


class TestModelKindDefinitions:
    """Verify model-kinds section of triforce.yaml."""

    def test_shared_nature_kind(self):
        """shared-nature should map to SharedNatureAgent."""
        mkd = ModelKindDefinition.get("shared-nature")
        assert mkd.network_class is SharedNatureAgent

    def test_multihead_kind(self):
        """multihead should map to MultiHeadAgent."""
        mkd = ModelKindDefinition.get("multihead")
        assert mkd.network_class is MultiHeadAgent

    def test_shared_nature_is_default(self):
        """shared-nature should be the default model kind."""
        mkd = ModelKindDefinition.get_default()
        assert mkd.name == "shared-nature"


class TestSkipSwordToTriforceScenario:
    """Verify skip-sword-to-triforce scenario end conditions for multihead training."""

    def test_scenario_exists(self):
        """skip-sword-to-triforce scenario should exist."""
        scenario = TrainingScenarioDefinition.get("skip-sword-to-triforce")
        assert scenario is not None, "skip-sword-to-triforce not found in triforce.yaml"

    def test_end_conditions_exact(self):
        """End conditions must include triforce collection and no early termination.

        No early termination conditions that would cap progress before triforce collection.
        """
        scenario = TrainingScenarioDefinition.get("skip-sword-to-triforce")
        expected = ["GainedTriforce", "GotWallmastered", "CollectedTreasure", "GameOver", "Timeout"]
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
