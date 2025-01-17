from typing import Dict

from .state_change_wrapper import StateChange
from .zelda_game import ZeldaGame
from .scenario_wrapper import TrainingScenarioDefinition
from . import critics
from . import end_conditions

def simulate_critique(env, action, scenario : TrainingScenarioDefinition, old : Dict, new : Dict):
    """Simulates the critic and end conditions for a scenario."""

    critic = getattr(critics, scenario.critic)()

    rewards = {}
    critic.clear()
    prev = ZeldaGame(env, old, 0)
    state = ZeldaGame(env, new, 0)
    change = StateChange(env, prev, state, action, [], {}, 0)
    critic.critique_gameplay(change, rewards)

    terminated = False
    truncated = False
    reason = None

    endings = [getattr(end_conditions, ec)() for ec in scenario.end_conditions]
    for ec in endings:
        ec.clear()
        terminated, truncated, reason = ec.is_scenario_ended(change)

    return rewards, terminated, truncated, reason
