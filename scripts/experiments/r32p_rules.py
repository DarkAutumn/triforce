"""Round 32's prerequisites: the `d3first` rooms where the walks' scripted rule beats the honest planner, as rules
over the environment's own actions; their won runs as demonstration rows (`train --demos`) and rule tracks
(docs/experiments/round32.md "Prerequisites"; the walks: docs/routes/d3first-walks-a.md).

    python scripts/experiments/r32p_rules.py ROOM [--runs N] [--chains 0,1,2] [--hearts own|family|chain|H]
                                                [--out FILE.npz] [--tracks DIR [--per-state N] [--max-loss H]]
                                                [--verbose]

ROOM plays its `d3first` route step from the route's installed states (`d3first_<stem>_<chain>`), N runs a state
(seeds 1000..), through the family's own environment; the play loop, the rows (a MOVE or WAIT cloned without its
duration, an item or sword row written several times), the tracks (`DIR/<state>~rule<seed>`, DIR is the family's
track directory) and the printed lines are round 29's (`r29_demos.play`), pointed at d3first's own steps.
`--hearts chain` draws every reset's hearts from the route's chain arrivals in the room (the family's config with
`ram_randomization.chain_hp: 1.0` and `state_routes: [d3first]`; `triforce/chain_hp.d3first.json`): the hearts the
route's chains bring.

ROOM is one of (the walks' rule each ports, `r32_d3first.LEGS`/`VARIANTS`):

    darknut-59    3:0x59 N->S (darknut, step 13): `BombFighter` (round 29's sword fighter with the Darknut's
                  shield; a bomb when a Darknut, moved on at its speed, will be within 10 px of the bomb's square
                  during the blast and the bomb does not meet its shield) until the room is all dead, then the S
                  shutter.  A bomb is the env's `BOMB <d>` (Link turns and lays it), not a turn and B.
    dash-69       3:0x69 N->cave (darknut, step 14): past the eight Darknuts without a stab (the fighter's floor
                  walk, keeping off what will touch Link) to ($C8,$8D), then E onto the stairs.
    manhandla-4d  3:0x4d W->N (boss-manhandla, step 22): `ManhandlaFighter` (a bomb where the base will be during
                  the blast; meanwhile in line with it 32-48 px away; the sword fight without bombs), the heart
                  container at ($78,$8D), the N door.
    gleeok-13     4:0x13 W->N (boss-gleeok, step 51): round 28's env Gleeok rule (`r28_demos.gleeok_rule`'s fight,
                  level 4's two-headed Gleeok, type $43), then the heart container at ($D0,$BD) and the N door.
    mouth-37      0:0x37 cave->cave (ow-octorok, step 62): round 31's mouth rule (`r31_mouth_demos.mouth_rule`).
    wall-2d       0:0x2d W->cave (ow-zora, step 88): to ($50,$5D) by the floor walk, BOMB N at the rock wall, N
                  into the cave.
    rupees-2d     0:0x2d's cave S->S (ow-zora, step 89): up to the 30 rupees at X $78, back, S out.
    candle-0c     0:0x0c's shop S->S (ow-tektite, step 94): up to the blue candle at X $98, back, S out.
    wrap-0c       0:0x0c cave->W (ow-tektite, step 95): to ($D8,$8D) by the floor walk (inside the wrap's zone, off
                  the E edge a Tektite's knock-back would push Link across), CLIP E (the world wrap W).
    scroll-29     0:0x29 W->E (ow-leever, step 82), scroll-2a 0:0x2a (83), scroll-2b 0:0x2b (84): the walks' screen
                  scroll (`CLIP W` on arrival at X 0: Link wraps to the room's E edge), then E out: 0 hearts.

The rooms the explorer does not solve from the route's states (rule tracks, `--tracks`), the walks' `clear` (the
sword fighter, `r32_walks.Fighter`, until the room is all dead) and then their way out:

    zol-6b        3:0x6b S->N (zol, step 8): ten Gels; the key at ($78,$8D), the N door.
    zol-4b        3:0x4b S->W (zol, step 10): three Zols; the key at ($78,$8D), the W door.
    like-like-32  4:0x32 W->cave (like-like, step 38): the Like-Likes; the block pushed E from ($50,$8D), the
                  stairs E of ($C8,$5D).
    vire-31       4:0x31 W->E (vire, step 37): the Vires; the E shutter.
    vire-12       4:0x12 N->E (vire, step 50): the Vire; the block pushed E from ($50,$8D), the E door."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pylint: disable=wrong-import-position
import r28_demos                                                  # noqa: E402
import r29_demos                                                  # noqa: E402
import r32_walks                                                  # noqa: E402  (round 29's Bot: overworld bounds)
from r29_demos import (BOMBS, LETTER, OBJ_TYPE, OBJ_X, OBJ_Y, PARALYZED, ROOM_ALL_DEAD, Decided, Way, action, go,
                       make_bot)                                  # noqa: E402

ROUTE = 'd3first'
RUPEES, INV_CANDLE, HEARTS, KEYS = 0x66D, 0x65B, 0x66F, 0x66E
GLEEOK2 = 0x43                                                    # level 4's two-headed Gleeok (r32_walks.GLEEOK2)

# room -> (family, d3first's route step)
ROOMS = {'darknut-59': ('darknut', 13), 'dash-69': ('darknut', 14), 'manhandla-4d': ('boss-manhandla', 22),
         'gleeok-13': ('boss-gleeok', 51), 'mouth-37': ('ow-octorok', 62), 'wall-2d': ('ow-zora', 88),
         'rupees-2d': ('ow-zora', 89), 'candle-0c': ('ow-tektite', 94), 'wrap-0c': ('ow-tektite', 95),
         'zol-6b': ('zol', 8), 'zol-4b': ('zol', 10), 'like-like-32': ('like-like', 38), 'vire-31': ('vire', 37),
         'vire-12': ('vire', 50), 'scroll-29': ('ow-leever', 82), 'scroll-2a': ('ow-leever', 83),
         'scroll-2b': ('ow-leever', 84)}


def route_step(index, route=ROUTE):
    """(the route's step `index`, the next step on its way): d3first's own numbers (`r28_demos.route_step` maps
    d2first's onto another route; these rooms are d3first's)."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    steps = entry_states.route_steps(route)
    way = (None, steps[index].way or 'clips')
    return steps[index], next((s for s in steps[index + 1:] if s.way in way), None)


def rule_env(family, state, hearts):
    """`r29_demos.rule_env` with the run-time clips switch on in every episode (d3first's steps 0-97 are the clip
    way, as the chains play them; a clip family's training draw would turn it off in half of them, and 0:0x0c's way
    out W is the E wrap), and `--hearts chain`: the family's draws with every reset's hearts from the route's chain
    arrivals in the room (`chain_hp: 1.0`), drawing d3first's states only."""
    from triforce._native import Env, Game                      # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist             # pylint: disable=import-outside-toplevel
    config = load_specialist(family)
    if config.to_dict()['clips']['enabled']:
        config = config.replace(clips={'enabled': True, 'off_share': 0.0})
    if hearts == 'own':
        config = config.replace(ram_randomization={})
    elif hearts == 'chain':
        spec = dict(config.to_dict()['ram_randomization'], chain_hp=1.0)
        config = config.replace(ram_randomization=spec, state_routes=[ROUTE])
    elif hearts != 'family':
        containers = Game(seed=0, states_dir=str(STATES_DIR)).load(state, fresh_room=True, seed=0).max_health
        share = float(hearts) / containers
        config = config.replace(ram_randomization={'hp': [share, share], 'full_hp': 0.0})
    return Env(config, explicit_starts=True, strict_actions=False, seed=0)


# ---- the fighters with bombs, asked once per action ------------------------------------------------------------------

class Bombed(Exception):
    """The fighter lays a bomb toward `button` (the env's `BOMB <d>`)."""

    def __init__(self, button):
        super().__init__(button)
        self.button = button


def act(bot, method, ok, leave=None):
    """`r29_demos.action`, with the fighter's bomb (`Decided(('BOMB', button))`) played as `BOMB <d>`."""
    def first():
        try:
            method()
        except Decided as d:
            if isinstance(d.choice, tuple) and d.choice[0] == 'BOMB':
                raise Bombed(d.choice[1]) from d
            raise
    try:
        return action(bot, first, ok, leave)
    except Bombed as b:
        return f'BOMB {LETTER[b.button]}'


def _spread(trail, x1, y1, n):
    """The frames between two decisions on a per-frame trail: n - 1 points from its last toward (x1, y1)."""
    if trail and n > 1:
        x0, y0 = trail[-1]
        for i in range(1, n):
            trail.append((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n))


def bomb_fighter():
    """`r32_walks.BombFighter` whose bomb is one decision (`BOMB <d>`, where the mask offers it)."""
    import r29_fight_walks as fw                                 # pylint: disable=import-outside-toplevel

    class EnvBombs(r32_walks.BombFighter):
        ok = staticmethod(lambda _step: False)

        def fill_frames(self, ram, n):
            for s, trail in self.trails.items():
                if ram[OBJ_TYPE + s] in self.bomb_types:
                    _spread(trail, ram[OBJ_X + s], ram[OBJ_Y + s], n)

        def fight_frame(self):
            w = self.w
            if self.can_act() and w.r(BOMBS) > self.reserve and not self.bomb_live():
                for button, (dx, dy) in r32_walks.AHEAD.items():
                    if not self.ok(f'BOMB {LETTER[button]}'):
                        continue
                    bx, by = w.x + 16 * dx, w.y + 16 * dy
                    for s, trail in self.trails.items():
                        if len(trail) < 4 or self.shielded(s, button):
                            continue
                        if any(abs(p[0] - bx) < 10 and abs(p[1] - by) < 10
                               for p in (self.ahead(s, k) for k in range(50, 70, 3))) and \
                                not self.danger((w.x, w.y), 4):
                            raise Decided(('BOMB', button))
            r32_walks.Fighter.fight_frame(self)
    assert fw.walk_graph is r32_walks.walk_graph
    return EnvBombs


def manhandla_fighter():
    """`r32_walks.ManhandlaFighter` whose bomb is one decision (`BOMB <d>`) and whose stand-off walk is the
    env bot's `toward` (it snaps Link to the grid)."""

    class EnvManhandla(r32_walks.ManhandlaFighter):
        ok = staticmethod(lambda _step: False)

        def fill_frames(self, ram, n):
            if ram[OBJ_TYPE + 5] == r32_walks.MANHANDLA:
                _spread(self.trail, ram[OBJ_X + 5], ram[OBJ_Y + 5], n)

        def fight_frame(self):
            w = self.w
            if not self.can_act() or not self.trail:
                return r32_walks.Fighter.fight_frame(self)
            if w.r(BOMBS) > 0 and not self.bomb_live():
                for button, (dx, dy) in r32_walks.AHEAD.items():
                    if not self.ok(f'BOMB {LETTER[button]}'):
                        continue
                    bx, by = w.x + 16 * dx, w.y + 16 * dy
                    if any(abs(p[0] - bx) < 10 and abs(p[1] - by) < 10
                           for p in (self.base_future(k) for k in range(50, 70, 3))) and not self.danger((w.x, w.y), 4):
                        raise Decided(('BOMB', button))
            if w.r(BOMBS) > 0 or self.bomb_live():
                self.try_stab()
                bx, by = self.trail[-1]
                goals = {n for n in self.graph if not self.danger(n) and
                         ((abs(n[0] - bx) <= 8 and 32 <= abs(n[1] - by) <= 48) or
                          (abs(n[1] - by) <= 8 and 32 <= abs(n[0] - bx) <= 48))}
                here = self.node()
                if here is not None and here in goals:
                    self.press(self.flee() if self.danger(here) else 'NONE')
                self.toward(goals)
            return r32_walks.Fighter.fight_frame(self)
    return EnvManhandla


def out(letter):
    """Through the doorway (or onto the stairs) toward `letter`."""
    def stage(_ram, ok):
        for step in (f'MOVE {letter} 8', f'MOVE {letter} 4'):
            if ok(step):
                return step
        return 'WAIT 4'
    return stage


def fight_rule(fighter, way_of):
    """The fighter (`fighter()` a class) until the room is all dead, then `way_of(bot)` (a `Way`)."""
    state = {}

    def rule(ram, ok):
        if 'bot' not in state:
            bot = state['bot'] = make_bot(fighter(), ram)
            if hasattr(bot, 'fill_frames'):        # `env_bot`'s own `fill` (a no-op) would shadow a base's
                bot.fill = bot.fill_frames
            state['way'] = way_of(state['bot'])
        bot = state['bot']
        bot.ok = ok
        bot.sync(ram)
        if not ram[ROOM_ALL_DEAD] and not state.get('left'):
            return act(bot, bot.fight_frame, ok)
        if not state.get('left'):
            state['left'] = True
            bot.targets = set()
        if ram[PARALYZED]:
            return act(bot, bot.fight_frame, ok)
        return state['way'].step(ram, ok)
    return rule


def walk_rule(way_of):
    """No fight: the floor walk of round 29's bot (keeping off what will touch Link), `way_of(bot)`."""
    state = {}

    def rule(ram, ok):
        if 'bot' not in state:
            state['bot'] = make_bot(r32_walks.Fighter, ram)
            state['way'] = way_of(state['bot'])
        state['bot'].sync(ram)
        return state['way'].step(ram, ok)
    return rule


def taken(address, test):
    """A stage done once `test(before, now)` holds of the byte at `address` (`before`: its value at the first ask)."""
    seen = {}

    def done(ram):
        return test(seen.setdefault('before', ram[address]), ram[address])
    return done


def take_stage(bot, node, done):
    """To `node` (the item's square) until `done(ram)`."""
    def stage(ram, ok):
        if done(ram):
            return None
        if (ram[OBJ_X], ram[OBJ_Y]) == node:
            return 'WAIT 4'
        return action(bot, lambda: bot.toward({node}), ok)
    return stage


def heart_way(node, door):
    """A boss room's end: the heart container at `node`, then the N door at `door`."""
    def way(bot):
        got = taken(HEARTS, lambda before, now: (now >> 4) > (before >> 4))
        return Way(take_stage(bot, node, got), go(bot, door, 'N'), out('N'))
    return way


def key_way(door, letter):
    """A key room's end: the key at ($78,$8D), then the door at `door` toward `letter`."""
    def way(bot):
        got = taken(KEYS, lambda before, now: now > before)
        return Way(take_stage(bot, (0x78, 0x8D), got), go(bot, door, letter), out(letter))
    return way


def push_way(stand, door, letter):
    """A block room's end: the block pushed E from `stand` (`r29_demos.push`), then the door or stairs at `door`
    toward `letter`."""
    def way(bot):
        return Way(r29_demos.push(bot, stand, 'E'), go(bot, door, letter), out(letter))
    return way


def cave_item_rule(x, done):
    """In a cave: up to the ware at X `x` until `done(ram)`, back to the entrance column, S out (the walks'
    `cave_item`)."""
    state = {}

    def rule(ram, ok):
        if 'x0' not in state:
            state['x0'] = ram[OBJ_X]
            state['fetch'] = r28_demos.Legs([('y', 0xBD), ('x', x)], 'MOVE N 4')
            state['back'] = r28_demos.Legs([('y', 0xBD), ('x', state['x0'])], 'MOVE S 8')
        if not state.get('got') and done(ram):
            state['got'] = True
        return (state['back'] if state.get('got') else state['fetch']).step(ram, ok)
    return rule


def gleeok_rule():
    """`r28_demos.gleeok_rule`'s fight on level 4's two-headed Gleeok, then the heart container at ($D0,$BD) (the
    walks' `OVERRIDES['4_13w']`) and the N door."""
    r28_demos.GLEEOK4 = GLEEOK2
    fight = r28_demos.gleeok_rule()
    got = taken(HEARTS, lambda before, now: (now >> 4) > (before >> 4))
    leave = r28_demos.Legs([('y', 0xBD), ('x', 0xD0), ('x', 0x78), ('y', 0x5D)], 'MOVE N 8', done={1: got})

    def rule(ram, ok):
        got(ram)
        if r28_demos.gleeok_heads(ram):
            return fight(ram, ok)
        return leave.step(ram, ok)
    return rule


def mouth_rule():
    import r31_mouth_demos                                       # pylint: disable=import-outside-toplevel
    return r31_mouth_demos.mouth_rule()


def scroll_rule():
    """The walks' screen scroll W -> E: `CLIP W` while Link stands on the W edge (the engine offers it there), once;
    then E out."""
    state = {}

    def rule(_ram, ok):
        if not state.get('clipped') and ok('CLIP W'):
            state['clipped'] = True
            return 'CLIP W'
        return out('E')(_ram, ok)
    return rule


RULES = {
    'darknut-59': lambda: fight_rule(bomb_fighter, lambda bot: Way(go(bot, (0x78, 0xBD), 'S'), out('S'))),
    'dash-69': lambda: walk_rule(lambda bot: Way(go(bot, (0xC8, 0x8D), 'E'), out('E'))),
    'manhandla-4d': lambda: fight_rule(manhandla_fighter, heart_way((0x78, 0x8D), (0x78, 0x5D))),
    'gleeok-13': gleeok_rule,
    'mouth-37': mouth_rule,
    'wall-2d': lambda: walk_rule(lambda bot: Way(go(bot, (0x50, 0x5D), 'N'), r29_demos.bomb_wall('N'))),
    'rupees-2d': lambda: cave_item_rule(0x78, taken(RUPEES, lambda before, now: now > before)),
    'candle-0c': lambda: cave_item_rule(0x98, taken(INV_CANDLE, lambda _before, now: now != 0)),
    'wrap-0c': lambda: walk_rule(lambda bot: Way(go(bot, (0xD8, 0x8D)),
                                                 lambda _ram, ok: 'CLIP E' if ok('CLIP E') else 'WAIT 4')),
    'zol-6b': lambda: fight_rule(lambda: r32_walks.Fighter, key_way((0x78, 0x5D), 'N')),
    'zol-4b': lambda: fight_rule(lambda: r32_walks.Fighter, key_way((0x20, 0x8D), 'W')),
    'like-like-32': lambda: fight_rule(lambda: r32_walks.Fighter, push_way((0x50, 0x8D), (0xC8, 0x5D), 'E')),
    'vire-31': lambda: fight_rule(lambda: r32_walks.Fighter, lambda bot: Way(go(bot, (0xD0, 0x8D), 'E'), out('E'))),
    'vire-12': lambda: fight_rule(lambda: r32_walks.Fighter, push_way((0x50, 0x8D), (0xD0, 0x8D), 'E')),
    'scroll-29': scroll_rule,
    'scroll-2a': scroll_rule,
    'scroll-2b': scroll_rule,
}


def main():
    r29_demos.ROOMS = ROOMS
    r29_demos.RULES = RULES
    r29_demos.rule_env = rule_env
    r28_demos.route_step = route_step
    if '--route' not in sys.argv:
        sys.argv += ['--route', ROUTE]
    return r29_demos.main()


if __name__ == '__main__':
    sys.exit(main())
