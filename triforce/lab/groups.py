"""The fresh generation's teachers as data (``triforce/groups.yaml``; docs/lab.md "Groups of families").

``load()`` reads every teacher: its members (policy keys ``family@route``, each a task of the lab trainer) and their
optional env shares; ``get(name)`` one of them.  ``python -m triforce.lab.train --group NAME`` trains it: its members
are the run's ``--families``, the file's route its ``--state-route``.  A malformed file is a ``ValueError`` naming
what to change: a member on another route or without one, a family in two teachers, a share of no member or not
above 0, a name a team spec cannot hold (``+``)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .. import registry
from ..paths import PACKAGE_DIR

GROUPS_FILE = PACKAGE_DIR / 'groups.yaml'


@dataclass(frozen=True)
class Group:
    """One teacher of the generation: ``members`` its policy keys in the file's order, ``shares`` member -> env
    share (absent: 1), ``route`` the states every member plays."""
    name: str
    route: str
    members: tuple
    shares: dict = field(default_factory=dict)

    @property
    def families(self) -> list[str]:
        return [registry.split_policy(key)[0] for key in self.members]

    def record(self) -> dict:
        """What a run's manifest names (``task.group``): the name, the route, the members and the shares."""
        return {'name': self.name, 'route': self.route, 'members': list(self.members), 'shares': dict(self.shares)}


def load(path: Optional[Path] = None) -> dict[str, Group]:
    """Every teacher of ``path`` (default ``GROUPS_FILE``), by name, checked (the module docstring)."""
    path = Path(path or GROUPS_FILE)
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    route = data.get('route')
    if not isinstance(route, str) or not route:
        raise ValueError(f"{path}: name the teachers' route (route: d3first)")
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
        unknown = sorted(set(entry) - {'members', 'shares'})
        if unknown:
            raise ValueError(f"{path}: teacher {name}: unknown keys {unknown} (members, shares)")
        groups[name] = Group(name, route, tuple(members), shares)
    return groups


def get(name: str, path: Optional[Path] = None) -> Group:
    """The teacher ``name`` (SystemExit listing the file's teachers when there is none)."""
    groups = load(path)
    if name not in groups:
        raise SystemExit(f"--group {name}: no such teacher in {path or GROUPS_FILE} (teachers: "
                         f"{', '.join(groups)})")
    return groups[name]
