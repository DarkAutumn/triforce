"""Network classes side by side (docs/lab.md).

A checkpoint names its architecture in ``model_config['arch']``; absent is today's ``model.ZeldaPolicy`` (``zelda``),
whose checkpoints stay as they were (no ``arch`` key, the Rust actor plays them). Every other architecture is a
``base.LabNet`` registered here by its ``arch`` and played through ``serve`` (no Rust port needed).

``build(model_config)`` rebuilds any of them; ``create(arch, obs_spec, **settings)`` makes a fresh one.
"""

from typing import Optional

from torch import nn

from ..model import ModelConfig, ZeldaPolicy
from .base import Heads, LabNet, factored_evaluate, factored_joint

ZELDA = 'zelda'
ARCHS: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator: a ``LabNet`` subclass playable by its ``arch``."""
    if not cls.arch or cls.arch in ARCHS or cls.arch == ZELDA:
        raise ValueError(f"{cls.__name__}: arch {cls.arch!r} is empty or taken")
    ARCHS[cls.arch] = cls
    return cls


def arch_of(model_config: Optional[dict]) -> str:
    """The architecture a ``model_config`` names (``zelda`` without ``arch``)."""
    return (model_config or {}).get('arch') or ZELDA


def names() -> list[str]:
    """Every architecture, ``zelda`` first."""
    return [ZELDA, *sorted(ARCHS)]


def build(model_config: dict) -> nn.Module:
    """The network a checkpoint's ``model_config`` describes, with fresh weights."""
    arch = arch_of(model_config)
    if arch == ZELDA:
        return ZeldaPolicy(config={k: v for k, v in model_config.items() if k != 'arch'})
    if arch not in ARCHS:
        raise ValueError(f"Unknown network arch {arch!r} (known: {names()})")
    return ARCHS[arch](model_config)


def create(arch: str, obs_spec: dict, **settings) -> nn.Module:
    """A fresh network of ``arch`` for ``obs_spec`` with ``settings`` over its defaults."""
    if arch == ZELDA:
        return ZeldaPolicy(config=ModelConfig.from_dict({**settings, 'obs_spec': obs_spec}))
    return build({**settings, 'arch': arch, 'obs_spec': obs_spec})


def is_native(network: nn.Module) -> bool:
    """Whether the Rust actor plays ``network`` (a ``ZeldaPolicy``); else it is served (``serve``)."""
    return isinstance(network, ZeldaPolicy)


# The architectures register on import.
from . import wide  # noqa: E402,F401  pylint: disable=wrong-import-position,cyclic-import
from . import gru, hithead, hitnet, roomheads  # noqa: E402,F401  pylint: disable=wrong-import-position,cyclic-import

__all__ = ['ARCHS', 'Heads', 'LabNet', 'ZELDA', 'arch_of', 'build', 'create', 'factored_evaluate', 'factored_joint',
           'is_native', 'names', 'register']
