from typing import Dict

from .game_state_change import ZeldaStateChange
from .zelda_game import ZeldaGame
from .models_and_scenarios import ZeldaScenario
from . import critics
from . import end_conditions

def simulate_critique(env, scenario : ZeldaScenario, old : Dict, new : Dict):
    """Simulates the critic and end conditions for a scenario."""

    critic = getattr(critics, scenario.critic)()

    rewards = {}
    critic.clear()
    prev = ZeldaGame(None, env, old, 0)
    state = ZeldaGame(prev, env, new, 0)
    change = ZeldaStateChange(env, prev, state, {})
    critic.critique_gameplay(change, rewards)

    terminated = False
    truncated = False
    reason = None

    endings = [getattr(end_conditions, ec)() for ec in scenario.end_conditions]
    for ec in endings:
        ec.clear()
        terminated, truncated, reason = ec.is_scenario_ended(change)

    return rewards, terminated, truncated, reason
