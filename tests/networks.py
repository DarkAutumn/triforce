"""Small networks with the native observation layout for tests that let the pool act: a random one and a scripted
one."""

import torch

from triforce import _native
from triforce.actions import Action
from triforce.model import ModelConfig, ZeldaPolicy


def native_spec() -> dict:
    """``ModelConfig.obs_spec`` for the native observation layout."""
    return {k: {'shape': list(v['shape']), 'vocab': v['vocab']} for k, v in _native.OBS_SPEC.items()}


def random_network(seed: int = 0, sharpen: float = 1.0) -> ZeldaPolicy:
    """Random weights, every parameter perturbed by 0.05·N(0, 1): a fresh network's biases are 0 and its LayerNorms
    1/0, so a native port that dropped any of them would still match it, and the perturbed heads (5× their init
    scale) make actions vary with the observation, not just the noise. ``sharpen`` scales the heads' weights further
    (logits grow, and so do float differences between torch and Rust)."""
    torch.manual_seed(seed)
    network = ZeldaPolicy(config=ModelConfig(obs_spec=native_spec())).eval()
    with torch.no_grad():
        for param in network.parameters():
            param.add_(0.05 * torch.randn_like(param))
        for head in (network.kind_head, network.direction_head, network.duration_head):
            head.weight.mul_(sharpen)
    return network


def scripted(action: str, fallback: str = 'WAIT 8') -> ZeldaPolicy:
    """A network that plays ``action`` where the mask allows it, else ``fallback`` (zeroed head weights, head
    biases 100 for ``action`` and 50 for ``fallback``)."""
    network = ZeldaPolicy(config=ModelConfig(obs_spec=native_spec()))
    heads = (network.kind_head, network.direction_head, network.duration_head)
    with torch.no_grad():
        for head in heads:
            head.weight.zero_()
            head.bias.zero_()
        for indices, bias in ((Action.parse(fallback).to_indices(), 50.0), (Action.parse(action).to_indices(), 100.0)):
            for head, index in zip(heads, indices):
                head.bias[int(index)] = bias
    return network
