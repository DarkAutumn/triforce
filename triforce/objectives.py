"""Objectives: where Link should go next (room exits, cave, treasure, enemies).

Ported from v1.  ``GameCompletion`` routes through the milestone: the sword first, then each triforce, using
``GameMap`` Dijkstra over game.yaml (key-aware) with ties broken toward the route (routes.yaml; ``milestone1`` by
default).  A route's ``waypoint:`` steps (the fairy pond on ``milestone1-fairy``) are visited first while they
apply.  Persistent RAM room flags (``room_state.room_flags``) seed which keys are already taken and which locked
doors are already open, so a start state in the middle of a dungeon routes correctly.  The objective targets feed
the (unobserved) wavefront used only for PBRS rewards; ``next_rooms`` decides which exits are correct.
"""

from dataclasses import dataclass
from enum import Enum

import inspect
import random
import sys
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

from .game_map import DEFAULT_ROUTE, GameMap, step_location
from .room_state import room_flags
from .zelda_game import ZeldaGame
from .zelda_enums import Direction, MapLocation, SwordKind, TileIndex, ZeldaItemKind

FLAG_ITEM_TAKEN = 0x10
# Pond fairy (Z_04 UpdatePondFairy): Link at the pond edge, Y=$AD with $70 <= X < $81, is halted until every heart
# is full.  Walking north from the south entry stops him there.
FAIRY_POND_TILES = (TileIndex(14, 14), TileIndex(15, 14), TileIndex(16, 14))


@dataclass(frozen=True)
class Waypoint:
    """A route stop off the shortest path (routes.yaml ``waypoint:``).  It applies while Link holds the triforce
    pieces the route took before it and not the next one; 'fairy' is done once he is in its room at full health."""
    step: int
    location: MapLocation
    kind: str
    after: FrozenSet[int]
    before: Optional[int]


def route_waypoints(steps: List[dict]) -> List[Waypoint]:
    """The waypoints of a route's steps, in route order."""
    result = []
    for index, step in enumerate(steps):
        if step.get('waypoint'):
            after = frozenset(s['level'] for s in steps[:index] if s['exit'] == 'triforce')
            before = next((s['level'] for s in steps[index:] if s['exit'] == 'triforce'), None)
            result.append(Waypoint(index, step_location(step), step['waypoint'], after, before))
    return result

_DIRECTION_DELTAS = [
    (Direction.E, (1, 0)),
    (Direction.W, (-1, 0)),
    (Direction.S, (0, 1)),
    (Direction.N, (0, -1)),
]

class ObjectiveKind(Enum):
    """The objective that the agent should be persuing."""
    NONE = 0
    MOVE = 1
    CAVE = 2
    TREASURE = 3
    FIGHT = 4
    ITEM = 5

class Objective:
    """The current objective for the agent."""
    def __init__(self, kind : ObjectiveKind, targets : Set[TileIndex], next_rooms : Set[int],
                 pbrs_targets : Set[TileIndex] = None):
        self.kind : ObjectiveKind = kind
        self.next_rooms : Sequence[MapLocation] = next_rooms if next_rooms is not None else set()
        self.targets : Sequence[TileIndex] = targets
        self.pbrs_targets : Sequence[TileIndex] = pbrs_targets if pbrs_targets is not None else targets

CAVE_TREASURE_TILE = TileIndex(0x0f, 0x0b)

class RoomMemory:
    """What the agent learns over the course of the playthrough."""
    def __init__(self, state : ZeldaGame, game_map : GameMap = None):
        self.level = state.level
        self.full_location = state.full_location
        self.exits = state.room.exits.copy()

        self.locked = None
        self.barred = None

        map_room = game_map.get(state.full_location) if game_map else None
        self.item = map_room.treasure if map_room else None
        if self.item is not None and state.level > 0 and room_flags(state, state.location) & FLAG_ITEM_TAKEN:
            self.item = None        # already taken on an earlier visit (persistent room flag, ram-catalog R.1)

        # Dungeon entrances have no south exit in the game map — block backtracking
        if map_room and state.level > 0 and 'S' not in map_room.exits:
            self.exits[Direction.S] = []

        self.referesh_state(state)


    def referesh_state(self, state):
        """On entry, refresh the state of the room's door locks."""
        self.locked = [x for x in self.exits if isinstance(x, Direction) and \
                       state.is_door_locked(x)]

        self.barred = [x for x in self.exits if isinstance(x, Direction) and \
                        state.is_door_barred(x)]

    def enumerate_adjacent_rooms(self):
        """Returns the list of all rooms connected to this one."""
        for direction in (x for x in self.exits if isinstance(x, Direction) and self.exits[x]):
            yield direction, self.full_location.get_location_in_direction(direction)

class ObjectiveSelector:
    """Determines the current objectives for the agent."""

    _game_map = None

    def __init__(self):
        self._rooms : Dict[RoomMemory] = {}
        shared_game_map()

    def get_current_objectives(self, prev : Optional[ZeldaGame], state : ZeldaGame) -> Objective:
        """Get the current objectives for the agent."""
        raise NotImplementedError()

    def _get_game_map(self) -> GameMap:
        """Returns the shared GameMap instance."""
        return ObjectiveSelector._game_map

    def _update_exits(self, state: ZeldaGame):
        if state.in_cave:
            return

        assert state.room.is_loaded
        if state.full_location not in self._rooms:
            room = RoomMemory(state, game_map=self._get_game_map())
            self._rooms[state.full_location] = room
        else:
            self._rooms[state.full_location].referesh_state(state)

    def _get_room(self, location):
        return self._rooms.get(location, None)

    def _get_enemy_tile_objectives(self, state):
        if state.active_enemies:
            return [tile for enemy in state.active_enemies for tile in enemy.link_overlap_tiles]

        return [tile for enemy in state.enemies for tile in enemy.link_overlap_tiles]


class _GameMapObjective(ObjectiveSelector):
    """Base class for objectives that use GameMap for routing."""

    def __init__(self, route: str = DEFAULT_ROUTE, collect_hearts: bool = False):
        super().__init__()
        if route not in self._game_map.routes:
            raise ValueError(f"Unknown route '{route}' (routes.yaml has {sorted(self._game_map.routes)})")
        self._route = route
        # Below max hearts, a dropped heart on the floor is the objective (ITEM) and the PBRS target.
        self.collect_hearts = collect_hearts
        self._last_route_key = None
        self._last_route_result = None
        self._collected_key_rooms: Set[MapLocation] = set()
        self._opened_doors: Set[tuple] = set()  # (MapLocation, direction_str) for doors opened at runtime
        self._seeded_location: Optional[MapLocation] = None

    _DIR_STR_TO_ENUM = {'N': Direction.N, 'S': Direction.S, 'E': Direction.E, 'W': Direction.W}

    def _update_opened_doors(self, state):
        """Detect doors that have been opened at runtime.

        Compares the static GameMap (which always marks doors as locked) against the
        actual tile state. When a door marked locked in the map is open in-game, it
        gets tracked so the Dijkstra treats it as free on future route calculations.
        """
        if state.level == 0:
            return

        if self._seeded_location != state.full_location:
            self._seeded_location = state.full_location
            self._seed_from_room_flags(state)

        map_room = self._game_map.get(state.full_location)
        if map_room is None:
            return

        for exit_info in map_room.exits.values():
            if exit_info.locked:
                door_key = (state.full_location, exit_info.direction)
                if door_key not in self._opened_doors:
                    direction = self._DIR_STR_TO_ENUM.get(exit_info.direction)
                    if direction and not state.is_door_locked(direction):
                        self._opened_doors.add(door_key)

    def _seed_from_room_flags(self, state):
        """Reads the level's persistent room flags: key rooms whose item is taken and locked doors already
        opened (flag bits E1 W2 S4 N8; opening sets both sides) count as collected/opened."""
        for loc, room in self._game_map.rooms.items():
            if loc.level != state.level or loc.in_cave:
                continue
            flags = room_flags(state, loc.value)
            if room.treasure == 'key' and flags & FLAG_ITEM_TAKEN:
                self._collected_key_rooms.add(loc)
            for exit_info in room.exits.values():
                direction = self._DIR_STR_TO_ENUM.get(exit_info.direction)
                if exit_info.locked and direction is not None and flags & direction.value:
                    self._opened_doors.add((loc, exit_info.direction))

    def _get_target(self, state: ZeldaGame) -> Optional[MapLocation]:
        """Return the MapLocation to route toward. Subclasses override."""
        raise NotImplementedError()

    def _should_collect_treasure(self, state: ZeldaGame) -> bool:  # pylint: disable=unused-argument
        """Whether to fight/collect treasure in the current room. Subclasses override.
        Used for caves only; dungeon rooms always collect."""
        return False

    def get_current_objectives(self, prev: Optional[ZeldaGame], state: ZeldaGame) -> Objective:
        self._update_exits(state)
        self._update_opened_doors(state)

        kind, tile_objectives, next_rooms = self._get_room_objective(prev, state)

        if kind == ObjectiveKind.NONE:
            kind = ObjectiveKind.MOVE

        pbrs_targets = set(tile_objectives)

        # Add dropped items to targets so wavefront lets link collect them
        for item in state.items:
            for tile in item.link_overlap_tiles:
                tile_objectives.append(tile)

        # Route to destination unless fighting/collecting/entering cave
        if kind not in (ObjectiveKind.FIGHT, ObjectiveKind.TREASURE, ObjectiveKind.CAVE):
            exit_tiles, rooms = self._get_map_objective(state)
            tile_objectives.extend(exit_tiles)
            next_rooms.extend(rooms)
            pbrs_targets.update(exit_tiles)

        if self.collect_hearts and state.link.health < state.link.max_health:
            hearts = {tile for item in state.items if item.id == ZeldaItemKind.Heart
                      for tile in item.link_overlap_tiles}
            if hearts:
                # Exits stay correct; only the kind and the shaping change (round 6: agents left half the hearts).
                return Objective(ObjectiveKind.ITEM, set(tile_objectives), set(next_rooms), hearts)
        return Objective(kind, set(tile_objectives), set(next_rooms), pbrs_targets)

    def _get_room_objective(self, prev, state):
        """Get in-room objectives: cave entry, dungeon fight/treasure."""
        tile_objectives = []
        next_rooms = []

        if state.in_cave:
            return self._get_cave_objective(state)

        if state.level > 0:
            return self._get_dungeon_room_objective(prev, state)

        # Overworld: check for cave entrance leading to our target
        cave_tile = state.room.cave_tile
        if cave_tile is not None:
            target = self._get_target(state)
            if target is not None:
                # Enter cave if target is inside this cave or is a dungeon entrance here
                cave_location = MapLocation(state.level, state.location, True)
                game_room = self._game_map.get(state.full_location)
                cave_exit = game_room.exits.get('cave') if game_room else None
                if cave_exit:
                    cave_dest = cave_exit.destination
                    # Enter if the cave IS the target, or leads to the target's level
                    if cave_dest == target or cave_dest.level == target.level:
                        # Dungeon entrance: cave leads to different level
                        if cave_dest.level != state.level:
                            next_rooms.append(cave_dest)
                        else:
                            next_rooms.append(cave_location)
                        tile_objectives.append(cave_tile)
                        return ObjectiveKind.CAVE, tile_objectives, next_rooms

        return ObjectiveKind.NONE, tile_objectives, next_rooms

    def _get_cave_objective(self, state):
        """Handle in-cave: collect treasure or leave."""
        tile_objectives = []
        next_rooms = []

        if self._should_collect_treasure(state):
            if state.treasure:
                tile_objectives.extend(state.treasure.link_overlap_tiles)
                return ObjectiveKind.TREASURE, tile_objectives, next_rooms

            tile_objectives.append(CAVE_TREASURE_TILE)
            return ObjectiveKind.TREASURE, tile_objectives, next_rooms

        # Leave the cave
        tile_objectives.extend(state.room.exits[Direction.S])
        next_rooms.append(MapLocation(state.level, state.location, False))
        return ObjectiveKind.MOVE, tile_objectives, next_rooms

    def _get_dungeon_room_objective(self, prev, state):
        """Handle dungeon room: always fight enemies / collect treasure when available."""
        kind = ObjectiveKind.NONE
        tile_objectives = []

        room_memory = self._rooms.get(state.full_location, None)
        if room_memory is None:
            return kind, tile_objectives, []

        # Track key collection: if treasure was just picked up, mark room as collected
        if prev and room_memory.item and prev.treasure is not None and state.treasure is None:
            if room_memory.item == "key":
                self._collected_key_rooms.add(state.full_location)
            room_memory.item = None

        # If we already collected this room's key, clear it so we don't fight/collect again
        if room_memory.item == 'key' and state.full_location in self._collected_key_rooms:
            room_memory.item = None

        # If treasure is dropped, collect it (skip map/compass)
        if state.treasure:
            if room_memory.item not in ("map", "compass"):
                kind = ObjectiveKind.TREASURE
                tile_objectives.extend(state.treasure.link_overlap_tiles)

        # If we know treasure exists but hasn't spawned, fight enemies to spawn it
        elif room_memory.item is not None and state.enemies:
            kind = ObjectiveKind.FIGHT
            tiles = self._get_enemy_tile_objectives(state)
            tile_objectives.extend(tiles)
            if not tiles:
                room_memory.item = None

        return kind, tile_objectives, []

    def _get_map_objective(self, state):
        """Route toward target using GameMap."""
        if state.in_cave:
            return state.room.exits[Direction.S], [MapLocation(state.level, state.location, False)]

        target = self._get_target(state)
        if target is None or state.full_location == target:
            return [], []

        # Cache to avoid re-routing every frame in the same room
        collected_frozen = frozenset(self._collected_key_rooms)
        opened_frozen = frozenset(self._opened_doors)
        route_key = (state.full_location, target, state.link.keys, collected_frozen, opened_frozen)
        if route_key == self._last_route_key:
            return self._last_route_result

        next_rooms = self._game_map.find_next_rooms(
            state.full_location, target, state.link.keys,
            collected_keys=self._collected_key_rooms,
            opened_doors=self._opened_doors
        )
        if not next_rooms:
            self._last_route_key = route_key
            self._last_route_result = ([], set())
            return [], set()

        # On the route (routes.yaml) only the route's exits are correct: ties between equally short paths go to the
        # route (to the exit it takes first when a room is visited more than once, e.g. 0x73 W before E), and a
        # shorter off-route path is not offered at all.  Rooms off the route keep plain shortest-path routing.
        route_next = self._game_map.route_next_rooms(state.full_location, self._route)
        if route_next:
            next_rooms = self._game_map.route_first(state.full_location, (next_rooms & route_next) or set(route_next),
                                                    self._route)

        directions = [state.full_location.get_direction_to(r) for r in next_rooms]
        targets = []
        for d, room in zip(directions, next_rooms):
            if d == Direction.NONE:
                # Cave transition: same location but different in_cave flag
                if room.in_cave != state.full_location.in_cave:
                    cave_tile = state.room.cave_tile
                    if cave_tile is not None:
                        targets.append(cave_tile)
                continue
            if d in state.room.exits:
                targets.extend(tiles_in_span(state.room.exits[d], exit_span(state.full_location, d), d))
            else:
                # Door exists (from routing) but isn't in exits — likely locked.
                # Add the Direction as a target so the wavefront can handle it
                # when locked_doors is set.
                targets.append(d)

        self._last_route_key = route_key
        self._last_route_result = (targets, next_rooms)
        return targets, next_rooms


class ReachLocation(_GameMapObjective):
    """Routes the agent to a specific map location."""
    def __init__(self, level=0, location=0, in_cave=False):
        super().__init__()
        loc = int(location, 16) if isinstance(location, str) else location
        self._target = MapLocation(level, loc, in_cave)

    def _get_target(self, state):
        return self._target

    def _should_collect_treasure(self, state):
        # Collect treasure if we're in the target room
        return state.full_location == self._target


class TreasureObjective(_GameMapObjective):
    """Routes the agent to collect a specific treasure."""
    def __init__(self, treasure="triforce"):
        super().__init__()
        self._treasure = treasure
        rooms = self._game_map.find_rooms_with_treasure(treasure)
        self._target_rooms = [r.location for r in rooms]

    def _get_target(self, state):
        if not self._target_rooms:
            return None

        # If we're already in one of the target rooms, stay
        for target in self._target_rooms:
            if state.full_location == target:
                return target

        # Find closest target room
        best = None
        best_cost = float('inf')
        for target in self._target_rooms:
            route = self._game_map.find_route(
                state.full_location, target, state.link.keys,
                collected_keys=self._collected_key_rooms,
                opened_doors=self._opened_doors
            )
            if route and len(route) < best_cost:
                best_cost = len(route)
                best = target

        return best

    def _should_collect_treasure(self, state):
        # Only fight/collect in rooms that have our target treasure
        return state.full_location in self._target_rooms


class GameCompletion(_GameMapObjective):
    """Routes the agent through the full game: swords first, then the route's waypoints while they apply, then
    triforce pieces.  ``step`` is the route step the episode starts at: waypoints before it are already done."""

    # Sword upgrade priority: (current_sword, treasure_name, min_health)
    _SWORD_UPGRADES = [
        (SwordKind.NONE, 'wood-sword', 0),
        (SwordKind.WOOD, 'white-sword', 5),
        (SwordKind.WHITE, 'magic-sword', 13),
    ]

    def __init__(self, route: str = DEFAULT_ROUTE, step: Optional[int] = None, collect_hearts: bool = False):
        super().__init__(route, collect_hearts)
        self._waypoints = [w for w in route_waypoints(self._game_map.routes[route]) if step is None or w.step >= step]

    def get_current_objectives(self, prev: Optional[ZeldaGame], state: ZeldaGame) -> Objective:
        waypoint = self._active_waypoint(state)
        if waypoint is not None and state.full_location == waypoint.location \
                and state.link.health >= state.link.max_health:
            self._waypoints.remove(waypoint)
        return super().get_current_objectives(prev, state)

    def _active_waypoint(self, state) -> Optional[Waypoint]:
        for waypoint in self._waypoints:
            if all(state.link.has_triforce(level) for level in waypoint.after) \
                    and (waypoint.before is None or not state.link.has_triforce(waypoint.before)):
                return waypoint
        return None

    def _get_room_objective(self, prev, state):
        waypoint = self._active_waypoint(state)
        if waypoint is not None and state.full_location == waypoint.location:
            return ObjectiveKind.TREASURE, list(FAIRY_POND_TILES), []
        return super()._get_room_objective(prev, state)

    def _get_target(self, state):
        waypoint = self._active_waypoint(state)
        return self._get_sword_target(state) or (waypoint.location if waypoint else None) \
            or self._get_triforce_target(state)

    def _get_sword_target(self, state):
        """If a better sword is available and Link meets the health requirement, target it."""
        for current_sword, treasure_name, min_health in self._SWORD_UPGRADES:
            if state.link.sword == current_sword and state.link.max_health >= min_health:
                rooms = self._game_map.find_rooms_with_treasure(treasure_name)
                if rooms:
                    return self._find_closest_room(state, [r.location for r in rooms])
            # Only check the first matching upgrade level
            if state.link.sword.value <= current_sword.value:
                break
        return None

    def _get_triforce_target(self, state):
        """Find the next triforce piece Link needs, routing to the closest reachable one."""
        triforce_rooms = self._game_map.find_rooms_with_treasure('triforce')
        # Filter to dungeons Link hasn't cleared yet
        targets = [r.location for r in triforce_rooms if not state.link.has_triforce(r.location.level)]
        if not targets:
            return None
        return self._find_closest_room(state, targets)

    def _find_closest_room(self, state, targets):
        """Find the closest reachable target room by Dijkstra cost."""
        best = None
        best_cost = float('inf')
        for target in targets:
            route = self._game_map.find_route(
                state.full_location, target, state.link.keys,
                collected_keys=self._collected_key_rooms,
                opened_doors=self._opened_doors
            )
            if route and len(route) < best_cost:
                best_cost = len(route)
                best = target
        return best

    def _should_collect_treasure(self, state):
        target = self._get_target(state)
        return target is not None and state.full_location == target


def shared_game_map() -> GameMap:
    """The GameMap every objective routes on (game.yaml + routes.yaml, loaded once per process)."""
    if ObjectiveSelector._game_map is None:
        ObjectiveSelector._game_map = GameMap.load()
    return ObjectiveSelector._game_map


def exit_span(location: MapLocation, direction: Direction) -> Optional[tuple]:
    """game.yaml ``span`` of `location`'s exit in `direction` (``RoomExit.span``), or None."""
    room = shared_game_map().get(location)
    exit_info = room.exits.get(direction.name) if room is not None else None
    return exit_info.span if exit_info is not None else None


def _along_edge(tile, direction: Direction) -> int:
    return tile.y if direction in (Direction.E, Direction.W) else tile.x


def tiles_in_span(tiles, span: Optional[tuple], direction: Direction) -> list:
    """The exit tiles inside `span` (all of them without one)."""
    if span is None:
        return list(tiles)
    return [tile for tile in tiles if span[0] <= _along_edge(tile, direction) <= span[1]]


def exit_is_correct(prev: ZeldaGame, location: MapLocation) -> bool:
    """Whether leaving `prev`'s room into `location` is a correct exit: the objective wanted that room and, if
    game.yaml limits the exit to a ``span`` (an edge whose openings lead into parts of the next room that don't
    connect, e.g. 0x5b E), Link left through it."""
    if location not in prev.objectives.next_rooms:
        return False
    direction = prev.full_location.get_direction_to(location)
    if direction not in (Direction.N, Direction.S, Direction.E, Direction.W):
        return True
    span = exit_span(prev.full_location, direction)
    return span is None or span[0] <= _along_edge(prev.link.tile, direction) <= span[1]


def start_route(location: MapLocation, entry: Optional[str], rng: Optional[random.Random] = None,
                route: Optional[str] = None) -> tuple:
    """(route, step) for an episode that starts in `location`, entered from `entry` (``start_states.state_entry``),
    for ``GameCompletion(route, step)``.  With `route` given: its first step there (None if it has none).  Else the
    routes entering that room from that side: if they all leave it the same way, the default route plays it;
    if not (0x38 from W: N on milestone1, S on milestone1-fairy), one of them at random, so training covers both."""
    game_map = shared_game_map()
    if route is not None:
        return route, game_map.route_step_index(route, location, entry) if entry else None
    candidates = game_map.routes_through(location, entry) if entry else []
    if not candidates:
        return DEFAULT_ROUTE, None
    if len({game_map.routes[name][index]['exit'] for name, index in candidates}) > 1:
        return (rng or random).choice(candidates)
    return next((c for c in candidates if c[0] == DEFAULT_ROUTE), candidates[0])


DUAL_EXIT_CHANCE = 0.05

class RoomWalk(ObjectiveSelector):
    """An objective selector that attempts to walk through rooms and battle enemies."""
    def __init__(self):
        super().__init__()
        self._curr_room = None
        self._target_exits = None
        self._next_rooms = None
        self._sequence = []

    def get_current_objectives(self, prev : Optional[ZeldaGame], state : ZeldaGame) -> Objective:
        self._update_exits(state)
        if prev is None:
            prev = state

        if self._curr_room != state.full_location:
            self._handle_room_change(prev, state)

        while True:
            objective = self._sequence[0]
            if objective.kind == ObjectiveKind.FIGHT:
                if state.enemies:
                    objective.targets = self._get_enemy_tile_objectives(state)
                else:
                    self._sequence.pop(0)
                    continue

            elif objective.kind == ObjectiveKind.TREASURE:
                if state.treasure is None:
                    self._sequence.pop(0)
                    continue

                objective.targets = list(state.treasure.link_overlap_tiles)

            return objective

    def _handle_room_change(self, prev, state):
        self._curr_room = state.full_location

        # have we already set the objectives?
        if self._sequence and state.full_location in self._sequence[0].next_rooms:
            self._sequence.pop(0)

            if self._sequence:
                return
        else:
            self._sequence.clear()

        room = state.room

        # Check if this room has a treasure
        room_memory = self._get_room(state.full_location)
        if room_memory is not None and room_memory.item is not None:
            self._sequence.append(Objective(ObjectiveKind.FIGHT, set(), set()))
            self._sequence.append(Objective(ObjectiveKind.TREASURE, set(), set()))

        # Check if this room has a cave.
        if cave := room.cave_tile:
            cave_location = MapLocation(state.level, state.location, True)
            self._sequence.append(Objective(ObjectiveKind.CAVE, [cave], [cave_location]))

            cave_south_exit = [TileIndex(x, 0x15) for x in range(0xe, 0x12)]
            self._sequence.append(Objective(ObjectiveKind.MOVE, cave_south_exit, [state.full_location]))

        valid_exits = [x for x in room.exits if isinstance(x, Direction) if state.is_door_open(x) and room.exits[x]]
        exits = list(valid_exits)
        came_from = self._came_from(prev, state)
        if came_from in exits:
            exits.remove(came_from)

        exits = self._get_reachable(exits, state)

        if len(exits) == 0:
            if came_from in valid_exits:
                exits.append(came_from)
            elif valid_exits:
                exits.append(random.choice(valid_exits))
            else:
                self._sequence.append(Objective(ObjectiveKind.MOVE, set(), set()))
                return

        if len(exits) == 1:
            self._target_exits = [exits[0]]

        else:
            count = 2 if random.random() < DUAL_EXIT_CHANCE else 1
            self._target_exits = random.sample(exits, k=count)

        next_rooms = set(state.full_location.get_location_in_direction(x) for x in self._target_exits)

        tile_objectives = [tile
                        for sublist in (state.room.exits[direction] for direction in self._target_exits)
                        for tile in sublist]

        self._sequence.append(Objective(ObjectiveKind.MOVE, set(tile_objectives), next_rooms))


    def _came_from(self, prev, state):
        came_from = state.full_location.get_direction_to(prev.full_location)
        if came_from != Direction.NONE:
            return came_from

        link = state.link
        distance_n = link.tile.y
        distance_s = state.room.tiles.shape[1] - link.tile.y
        distance_w = link.tile.x
        distance_e = state.room.tiles.shape[0] - link.tile.x

        distance = min(distance_n, distance_s, distance_w, distance_e)
        if distance == distance_n:
            return Direction.N
        if distance == distance_s:
            return Direction.S
        if distance == distance_w:
            return Direction.W
        return Direction.E

    def _get_reachable(self, exits, state):
        exit_tiles = {}
        for direction in exits:
            room = self._get_room(state.full_location)
            for tile in room.exits[direction]:
                exit_tiles[tile] = direction

        room = state.room
        result = set()
        seen = set()
        todo = [state.link.tile]
        while todo:
            tile = todo.pop()
            if tile in seen:
                continue
            seen.add(tile)

            if tile in exit_tiles:
                result.add(exit_tiles[tile])

            for direction, (dx, dy) in _DIRECTION_DELTAS:
                next_tile = TileIndex(tile.x + dx, tile.y + dy)
                if 0 <= next_tile.x < room.tiles.shape[0] and 0 <= next_tile.y < room.tiles.shape[1]:
                    if next_tile not in seen and room.can_move(tile.x, tile.y, direction):
                        todo.append(next_tile)

        return list(result)

def _init_objectives():
    # Get all classes defined in this module
    result = {}
    current_module = sys.modules[__name__]
    for cls_name, cls_obj in inspect.getmembers(current_module, inspect.isclass):
        if issubclass(cls_obj, ObjectiveSelector) and cls_obj is not ObjectiveSelector \
                and not cls_name.startswith('_'):
            result[cls_name] = cls_obj

    return result

OBJECTIVES = _init_objectives()

def get_objective_selector(name):
    """Get the objective by name."""
    return OBJECTIVES[name]

__all__ = [
    Objective.__name__,
    GameCompletion.__name__,
    ReachLocation.__name__,
    TreasureObjective.__name__,
    ObjectiveKind.__name__,
    get_objective_selector.__name__,
]
