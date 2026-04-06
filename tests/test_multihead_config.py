"""Tests for triforce.yaml config: action spaces, model kinds, and scenario wiring.

Verifies:
- Action space definitions load correctly from triforce.yaml
- Model kind definitions load correctly and map to the right classes
- Defaults are set correctly
- GameOver is always added to end conditions
- dungeon1 scenario has correct end conditions
"""

from triforce.models import ActionSpaceDefinition, ModelKindDefinition, MultiHeadAgent, SharedNatureAgent, \
    ImpalaMultiHeadAgent
from triforce.scenario_wrapper import TrainingScenarioDefinition


class TestActionSpaceDefinitions:
    """Verify action-spaces section of triforce.yaml."""

    def test_sword_only_action_space(self):
        """sword-only action space should have MOVE, SWORD, BEAMS."""
        asd = ActionSpaceDefinition.get("sword-only")
        assert asd.actions == ["MOVE", "SWORD", "BEAMS"]

    def test_sword_only_is_default(self):
        """sword-only should be the default action space."""
        asd = ActionSpaceDefinition.get_default()
        assert asd.name == "sword-only"

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

    def test_impala_multihead_is_default(self):
        """impala-multihead should be the default model kind."""
        mkd = ModelKindDefinition.get_default()
        assert mkd.name == "impala-multihead"


class TestScenarioDefaults:
    """Verify default values and GameOver injection."""

    def test_gameover_always_present(self):
        """GameOver should be in end conditions even if not specified."""
        scenario = TrainingScenarioDefinition.get("full-game")
        assert "GameOver" in scenario.end_conditions

    def test_default_critic(self):
        """Scenarios without explicit critic should default to GameplayCritic."""
        scenario = TrainingScenarioDefinition.get("full-game")
        assert scenario.critic == "GameplayCritic"

    def test_default_scenario_selector(self):
        """Scenarios without explicit selector should default to round-robin."""
        scenario = TrainingScenarioDefinition.get("full-game")
        assert scenario.scenario_selector == "round-robin"

    def test_structured_objective(self):
        """Scenarios with structured objectives should parse kind and params."""
        scenario = TrainingScenarioDefinition.get("overworld-skip-sword")
        obj_class, obj_params = scenario.objective
        assert obj_class.__name__ == "ReachLocation"
        assert obj_params["level"] == 1

    def test_default_objective(self):
        """Scenarios without explicit objective should default to GameCompletion."""
        scenario = TrainingScenarioDefinition.get("full-game")
        obj_class, obj_params = scenario.objective
        assert obj_class.__name__ == "GameCompletion"
        assert obj_params == {}


class TestDungeon1Scenario:
    """Verify dungeon1 scenario configuration."""

    def test_scenario_exists(self):
        """dungeon1 scenario should exist."""
        scenario = TrainingScenarioDefinition.get("dungeon1")
        assert scenario is not None, "dungeon1 not found in triforce.yaml"

    def test_end_conditions(self):
        """End conditions should include triforce collection and GameOver."""
        scenario = TrainingScenarioDefinition.get("dungeon1")
        assert "GainedTriforce" in scenario.end_conditions
        assert "GameOver" in scenario.end_conditions
        assert "Timeout" in scenario.end_conditions

    def test_no_missing_end_conditions(self):
        """Scenario should have all expected dungeon end conditions."""
        scenario = TrainingScenarioDefinition.get("dungeon1")
        expected = {"LeftDungeon", "GainedTriforce", "CollectedTreasure",
                    "Timeout", "Dungeon1DidntGetKey", "NowhereToGoCondition", "GameOver"}
        assert set(scenario.end_conditions) == expected
