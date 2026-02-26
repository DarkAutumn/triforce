"""Exploration objective system analysis.

Traces through the objective system to show how next_rooms are computed,
how the A* pathfinding works, and how location rewards interact with
objectives. Analyzes the objective graph for both overworld and dungeon.

Does NOT require the NES ROM.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from triforce.objectives import (
    GameCompletion, RoomWalk, ObjectiveSelector, ObjectiveKind,
    overworld_to_item, dungeon_to_item, dungeon_entrances,
    item_to_overworld, item_to_dungeon, LOCKED_DISTANCE
)
from triforce.zelda_enums import MapLocation, SwordKind, Direction, ZeldaItemKind


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_overworld_routing():
    """Show the overworld routing decisions."""
    print_header("Overworld Routing Decisions (GameCompletion._get_map_objective)")
    print()
    print("  The routing logic is a series of if/elif conditions:")
    print()
    print("  1. sword == NONE → go to 0x77 (sword cave)")
    target = item_to_overworld.get(SwordKind.WOOD, 'NOT_FOUND')
    print(f"     target_location = item_to_overworld[SwordKind.WOOD] = 0x{target:02x}")
    print()
    print("  2. sword == WOOD and max_health >= 5 → go to white sword")
    target = item_to_overworld.get(SwordKind.WHITE, 'NOT_FOUND')
    print(f"     target_location = item_to_overworld[SwordKind.WHITE] = 0x{target:02x}")
    print()
    print("  3. max_health >= 13 → go to magic sword (unreachable in current training)")
    print()
    print("  4. triforce_pieces == 0 → go to dungeon 1 (with dungeon 2 fallback)")
    for d, loc in dungeon_entrances.items():
        ow = item_to_overworld.get(d, 'N/A')
        print(f"     dungeon {d}: overworld location 0x{ow:02x}, dungeon entrance 0x{loc:02x}")
    print()
    print("  5. else → go to next uncleared dungeon")

    print_header("Dungeon Routing Decisions (GameCompletion._get_map_objective)")
    print()
    print("  Directional hints (hardcoded):")
    print("    - At (1, 0x53): → go to (1, 0x52)")
    print("    - At (1, 0x52): → go to (1, 0x42)")
    print("    - At (1, 0x43) with keys: → go to (1, 0x44) via E door")
    print("    - At (1, 0x45) with keys: → go to (1, 0x35) via N door")
    print("    - Otherwise: A* to triforce room")

    for room, item in dungeon_to_item.items():
        print(f"    dungeon room 0x{room:02x}: {item}")


def analyze_location_tracking():
    """Analyze how _correct_locations is tracked."""
    print_header("Location Tracking in GameplayCritic.critique_location_change")
    print()
    print("  State tracking:")
    print("    _correct_locations : set  — tracks visited rooms")
    print()
    print("  Flow (critics.py lines 294-315):")
    print("    1. prev = state_change.previous.full_location  (MapLocation)")
    print("    2. curr = state_change.state.full_location  (MapLocation)")
    print("    3. If gained_triforce → skip (no reward/penalty)")
    print("    4. FIRST MOVE SEED (lines 303-305):")
    print("       if prev != curr and not self._correct_locations:")
    print("           self._correct_locations.add((prev, curr))  # adds TUPLE!")
    print()
    print("  BUG: The seed adds a (MapLocation, MapLocation) TUPLE to the set.")
    print("  But line 309 checks 'if curr in self._correct_locations' where curr")
    print("  is a MapLocation. A MapLocation will never == a tuple, so the seed")
    print("  serves no purpose. The immunity described in the todo is illusory.")
    print()
    print("    5. If curr is in objectives.next_rooms:")
    print("       - If curr already in _correct_locations → revisit (+0.05)")
    print("       - Else → new location (+0.75), add curr to _correct_locations")
    print("    6. Else → wrong location (-1.00)")
    print()
    print("  CRITICAL: The only way to avoid the -1.0 penalty is if curr is in")
    print("  state_change.previous.objectives.next_rooms. This is set by")
    print("  ObjectiveSelector and depends on A* pathfinding.")


def analyze_cave_transitions():
    """Analyze cave transition handling in OverworldSwordCritic."""
    print_header("OverworldSwordCritic Cave Transition Logic")
    print()
    print("  Location 0x77 is the starting room with the sword cave.")
    print()
    print("  Cave transitions (critics.py lines 450-472):")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │ State Transition              │ Sword? │ Reward     │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │ Not-cave → Cave               │ Yes    │ -1.00 (reentered) │")
    print("  │ Not-cave → Cave               │ No     │ +0.75 (entered)   │")
    print("  │ Cave → Not-cave               │ Yes    │ +0.75 (left ok)   │")
    print("  │ Cave → Not-cave               │ No     │ -1.00 (left early)│")
    print("  │ Not-cave → Not-cave (!=0x77)  │ Yes    │ +0.75 (new loc)   │")
    print("  │ Not-cave → Not-cave (!=0x77)  │ No     │ -0.75 (left area) │")
    print("  │ Not-cave → Not-cave (==0x77)  │ Any    │ nothing            │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  KEY ISSUE: 'Not-cave → Not-cave with sword' always gives +0.75")
    print("  regardless of direction! If the agent wanders randomly after getting")
    print("  the sword, every room transition is rewarded.")
    print()
    print("  This critic COMPLETELY REPLACES GameplayCritic.critique_location_change.")
    print("  It does NOT check objectives or A* routing at all.")
    print("  It only fires for the 'overworld-sword' scenario (starting room only).")


def analyze_objective_interaction():
    """Show how objectives interact with location rewards."""
    print_header("Objectives ↔ Location Rewards Interaction")
    print()
    print("  The chain is:")
    print("  1. StateChangeWrapper._update_state calls ObjectiveSelector.get_current_objectives")
    print("  2. Objectives are stored as state.objectives (on the CURRENT state)")
    print("  3. critique_location_change reads state_change.previous.objectives.next_rooms")
    print()
    print("  So the 'allowed rooms' check uses the PREVIOUS state's objectives.")
    print("  This means: the room the agent just left determines what rooms are valid.")
    print()
    print("  If the agent is in room A and objective says 'go to B or C',")
    print("  then entering B is rewarded and entering D is penalized.")
    print()
    print("  PROBLEM: The A* pathfinder may not find the optimal path, or may find")
    print("  multiple paths and not include a valid one. If the map memory is wrong,")
    print("  the next_rooms set could be wrong, causing valid transitions to be penalized.")
    print()
    print("  ALSO: next_rooms is ONLY populated when objective.kind is not FIGHT,")
    print("  TREASURE, or CAVE (objectives.py line 243). During combat or treasure")
    print("  collection, next_rooms is empty → any room transition is penalty!")

    print_header("RoomWalk Objective: Single-Room Training")
    print()
    print("  RoomWalk randomly selects 1-2 exit directions as targets.")
    print("  next_rooms is the set of rooms reached by those exits.")
    print("  If the agent takes the 'wrong' exit, it gets -1.0.")
    print()
    print("  With DUAL_EXIT_CHANCE = 0.05, 95% of the time only one exit is valid.")
    print("  In a 4-exit room, the agent has a 75% chance of picking the wrong one!")
    print()
    print("  The penalty is MAXIMUM (-1.0), same as being wallmastered.")
    print("  This trains the agent to be terrified of room transitions.")


def analyze_end_conditions():
    """Show how end conditions interact with exploration."""
    print_header("End Conditions Related to Exploration")
    print()
    print("  Timeout (end_conditions.py):")
    print(f"    position_timeout = 50 steps stuck in one position → truncated")
    print(f"    no_discovery_timeout = 1200 steps without new room → truncated")
    print(f"    no_next_room_timeout = 300 steps without correct room → truncated")
    print(f"    tile_timeout = 30 steps on same tile → truncated")
    print()
    print("  BUG in Timeout.is_scenario_ended line 71:")
    print("    self.__seen.add(curr.location)  ← adds .location not .full_location!")
    print("    This means cave visits don't count as new discoveries if the base")
    print("    location was already seen.")
    print()
    print("  RoomWalkCondition (end_conditions.py):")
    print(f"    same_room_timeout = 750 steps → truncated")
    print("    On room change: success if in next_rooms, failure-wrong-exit otherwise")
    print("    This TERMINATES on any room change, not just wrong ones!")
    print()
    print("  The interplay:")
    print("    1. Agent can't find the right exit → stuck penalty (-0.01*N)")
    print("    2. Agent exits wrong way → -1.0 AND terminated")
    print("    3. Agent takes too long → truncated (no extra penalty)")
    print("    4. Right exit → only +0.75 for new, +0.05 for revisit")
    print()
    print("  Expected value analysis (random agent in 4-exit room, 1 correct):")
    print("    P(correct) = 0.25, reward = +0.75")
    print("    P(wrong)   = 0.75, reward = -1.00")
    print(f"    E[reward]  = 0.25 * 0.75 + 0.75 * -1.00 = {0.25*0.75 + 0.75*-1.0:.4f}")
    print("    A random agent sees strongly negative exploration reward.")


def main():
    analyze_overworld_routing()
    analyze_location_tracking()
    analyze_cave_transitions()
    analyze_objective_interaction()
    analyze_end_conditions()

    print_header("Summary of Issues Found")
    print("""
  1. PENALTY_WRONG_LOCATION = -1.0 (MAXIMUM) — same as wallmaster/death
  2. REWARD_REVIST_LOCATION = +0.05 (TINY) — barely distinguishable from noise
  3. Dead code: lines 303-305 add tuple to set, never matches location check
  4. OverworldSwordCritic rewards ALL room transitions with sword (+0.75)
  5. No inter-room PBRS — no signal between room transitions
  6. RoomWalk 75% chance of -1.0 penalty in 4-exit room
  7. Timeout bug: .location vs .full_location in discovery tracking
  8. next_rooms empty during FIGHT/TREASURE/CAVE → any transition = -1.0
  9. TrainingHints prevents wrong exits but doesn't help with exploration signal
  10. End conditions terminate on wrong exit, compounding the -1.0 penalty
""")


if __name__ == '__main__':
    main()
