from typing import Dict, Tuple
from .models_and_scenarios import ZeldaScenario
from . import critics
from . import end_conditions

def simulate_critique(scenario : ZeldaScenario, old : Dict, new : Dict) -> Tuple[Dict[str, float], bool, bool, str]:
    """Simulates the critic and end conditions for a scenario."""

    critic = getattr(critics, scenario.critic)()

    rewards = {}
    critic.clear()
    critic.critique_gameplay(old, new, rewards)

    terminated = False
    truncated = False
    reason = None

    endings = [getattr(end_conditions, ec)() for ec in scenario.end_conditions]
    for ec in endings:
        ec.clear()
        terminated, truncated, reason = ec.is_scenario_ended(old, new)

    return rewards, terminated, truncated, reason
