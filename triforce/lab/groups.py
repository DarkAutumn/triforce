"""The fresh generation's teachers as data (``triforce/groups.yaml``; docs/lab.md "Groups of families").

``load()`` reads every teacher: its members (policy keys ``family@route``, each a task of the lab trainer), their
optional env shares and the network the generation trains (the file's ``network:`` block, a teacher's own
``network:`` keys over it); ``get(name)`` one of them.  ``python -m triforce.lab.train --group NAME`` trains it: its
members are the run's ``--families``, the file's route its ``--state-route``, its network the run's ``--arch``,
``--arch-config``, ``--aux`` and ``--try-others`` unless the command line gives them.  A malformed file is a
``ValueError`` naming what to change: a member on another route or without one, a family in two teachers, a share of
no member or not above 0, a name a team spec cannot hold (``+``), a network key the lab has no option for."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .. import registry
from ..paths import PACKAGE_DIR

GROUPS_FILE = PACKAGE_DIR / 'groups.yaml'
# What a ``network:`` block may pin (the lab's ``--arch``, ``--arch-config``, ``--aux``, ``--try-others``).
NETWORK_KEYS = ('arch', 'arch_config', 'aux', 'try_others')


@dataclass(frozen=True)
class Group:
    """One teacher of the generation: ``members`` its policy keys in the file's order, ``shares`` member -> env
    share (absent: 1), ``route`` the states every member plays."""
    name: str
    route: str
    members: tuple
    shares: dict = field(default_factory=dict)
    # The network it trains (``NETWORK_KEYS``: the file's ``network:`` block, the teacher's own keys over it; an
    # ``arch_config`` merged by key); empty: the file pins none.
    network: dict = field(default_factory=dict)

    @property
    def families(self) -> list[str]:
        return [registry.split_policy(key)[0] for key in self.members]

    def record(self) -> dict:
        """What a run's manifest names (``task.group``): the name, the route, the members, the shares and, when the
        file pins one, the network."""
        out = {'name': self.name, 'route': self.route, 'members': list(self.members), 'shares': dict(self.shares)}
        if self.network:
            out['network'] = dict(self.network)
        return out


def load(path: Optional[Path] = None) -> dict[str, Group]:
    """Every teacher of ``path`` (default ``GROUPS_FILE``), by name, checked (the module docstring)."""
    path = Path(path or GROUPS_FILE)
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    route = data.get('route')
    if not isinstance(route, str) or not route:
        raise ValueError(f"{path}: name the teachers' route (route: d3first)")
    generation = _network(data.get('network'), f"{path}: network")
    groups: dict[str, Group] = {}
    owner: dict[str, str] = {}
    for name, entry in (data.get('teachers') or {}).items():
        name = str(name)
        if '+' in name or '/' in name or not name:
            raise ValueError(f"{path}: teacher {name!r}: a name without '+' or '/' (a team spec splits on them)")
        entry = entry or {}
        members = [str(m) for m in entry.get('members') or ()]
        if not members:
            raise ValueError(f"{path}: teacher {name} has no members")
        for key in members:
            family, member_route = registry.split_policy(key)
            if member_route != route:
                raise ValueError(f"{path}: teacher {name}: member {key} is not on route {route} "
                                 f"(write it {family}@{route})")
            if family in owner:
                raise ValueError(f"{path}: {family} is in two teachers, {owner[family]} and {name}: one teacher a "
                                 f"family")
            owner[family] = name
        shares = {str(k): float(v) for k, v in (entry.get('shares') or {}).items()}
        bad = sorted(k for k, v in shares.items() if k not in members or not v > 0)
        if bad:
            raise ValueError(f"{path}: teacher {name}: shares {bad} name no member or are not above 0")
        unknown = sorted(set(entry) - {'members', 'shares', 'network'})
        if unknown:
            raise ValueError(f"{path}: teacher {name}: unknown keys {unknown} (members, shares, network)")
        own = _network(entry.get('network'), f"{path}: teacher {name}")
        merged = {**generation, **own}
        if own.get('arch', generation.get('arch')) != generation.get('arch'):
            merged['arch_config'] = dict(own.get('arch_config') or {})     # another architecture: none of the file's
        else:
            merged['arch_config'] = {**(generation.get('arch_config') or {}), **(own.get('arch_config') or {})}
        if not merged['arch_config'] and 'arch_config' not in generation and 'arch_config' not in own:
            del merged['arch_config']
        groups[name] = Group(name, route, tuple(members), shares, merged)
    return groups


def _network(block, where: str) -> dict:
    """A ``network:`` block checked (``NETWORK_KEYS``; ``arch_config`` a mapping, ``aux`` a list, ``try_others`` K:P)."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{where}: network is a mapping of {', '.join(NETWORK_KEYS)}")
    unknown = sorted(set(block) - set(NETWORK_KEYS))
    if unknown:
        raise ValueError(f"{where}: network: unknown keys {unknown} ({', '.join(NETWORK_KEYS)})")
    out = dict(block)
    if 'arch' in out and not isinstance(out['arch'], str):
        raise ValueError(f"{where}: network.arch is an architecture's name (hitnet)")
    if 'arch_config' in out and not isinstance(out['arch_config'], dict):
        raise ValueError(f"{where}: network.arch_config is a mapping (hidden_dim: 768)")
    if 'aux' in out:
        if not isinstance(out['aux'], list) or not all(isinstance(a, str) for a in out['aux']):
            raise ValueError(f"{where}: network.aux is a list of OUTPUT=TARGET:COEFF ([hit=hit:0.5])")
    if 'try_others' in out and out['try_others'] is not None:
        out['try_others'] = str(out['try_others'])
    return out


def get(name: str, path: Optional[Path] = None) -> Group:
    """The teacher ``name`` (SystemExit listing the file's teachers when there is none)."""
    groups = load(path)
    if name not in groups:
        raise SystemExit(f"--group {name}: no such teacher in {path or GROUPS_FILE} (teachers: "
                         f"{', '.join(groups)})")
    return groups[name]
