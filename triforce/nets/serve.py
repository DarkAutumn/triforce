"""Serving torch networks to the pool's workers (docs/lab.md "How an experimental network acts").

A ``Server`` owns one ``_native.ExternalQueue`` and a thread that loops: take every pending observation
(``ExternalQueue.take``), run each network on its rows (``LabNet.heads``; any module with that method, ``ZeldaPolicy``
too), answer the logits and values. The workers pick the actions in Rust from those logits, by the act mode, with
the keyed noise (``policy::logits``). ``Server.policy(network)`` is the network as a policy for ``EnvPool.play`` /
``EnvPool.rollout`` (beside Rust ``_native.Policy`` objects in the same call).

Recurrent networks: the state of each env (the queue's ``slot``) is kept here and starts over at a job's step 0 or a
new job; a bootstrap request (a truncation's value) reads the state without advancing it.

The thread runs while the process lives (a daemon); a failure in a network closes the queue, so every waiting env
fails with the error instead of hanging.
"""
import copy
import hashlib
import json
import os
import threading
import time
from typing import Optional

import numpy as np
import torch

from .. import _native

DEFAULT_GATHER = 0.005      # at most this many seconds a batch waits for the other active envs' rows
ACTIVE_SECONDS = 1.0        # an env that asked this recently is waited for (``Server._loop``)
CPU_THREADS = 4             # torch threads when serving on the CPU


def serve_device() -> torch.device:
    """``$TRIFORCE_SERVE_DEVICE`` (cpu | cuda), else cuda when available."""
    name = os.environ.get('TRIFORCE_SERVE_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def weights_hash(network: torch.nn.Module) -> int:
    """64 bits identifying the architecture and the weights (records' model hash for a served network)."""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(json.dumps(network.config.to_dict(), sort_keys=True, default=str).encode())
    for name, value in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().to('cpu', torch.float32).contiguous().numpy().tobytes())
    return int.from_bytes(digest.digest(), 'little')


class Server:
    """Serves networks to pools through one queue (the module docstring)."""

    def __init__(self, device=None, gather: float = DEFAULT_GATHER, threads: int = CPU_THREADS):
        self.device = torch.device(device) if device is not None else serve_device()
        self.gather = gather
        if self.device.type == 'cpu':
            # torch's default (every core) contends with the pool's workers: measured 4 decisions/s on 10 envs.
            torch.set_num_threads(threads)
        self._seen: dict[int, float] = {}       # slot -> when it last asked (the envs a batch waits for)
        self.queue = _native.ExternalQueue()
        self.models: dict[int, torch.nn.Module] = {}
        self._keys: dict = {}
        self._states: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None
        self.batches = 0
        self.rows = 0

    def policy(self, network: torch.nn.Module, key=None):
        """``network`` (moved to the server's device, eval mode) as an ``_native.ExternalPolicy``.  The same
        ``key`` (default: the network object) reuses its model index and replaces the network there."""
        key = id(network) if key is None else key
        with self._lock:
            index = self._keys.setdefault(key, len(self._keys))
            self.models[index] = network.to(self.device).eval()
            self._states = {k: v for k, v in self._states.items() if k[0] != index}
        self._start()
        return self.queue.policy(index, weights_hash(network))

    def serve_copy(self, network: torch.nn.Module, key=None):
        """``policy`` of a frozen copy of ``network`` (``key`` default: the network object, so a training run's
        rollouts reuse one index and the copy before is dropped)."""
        return self.policy(copy.deepcopy(network), id(network) if key is None else key)

    def _start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name='net-server', daemon=True)
            self._thread.start()

    def _loop(self):
        while not self.queue.closed:
            # Wait for every env that asked in the last second (they step their emulators at different times; a
            # batch per straggler costs a forward each), at most ``gather`` seconds.
            now = time.monotonic()
            self._seen = {slot: t for slot, t in self._seen.items() if now - t < ACTIVE_SECONDS}
            batch = self.queue.take(timeout=0.1, gather=self.gather, target=max(1, len(self._seen)))
            if batch is None:
                continue
            now = time.monotonic()
            for slot in batch['slot']:
                self._seen[int(slot)] = now
            try:
                self.answer(batch)
            except BaseException as error:  # pylint: disable=broad-exception-caught
                self.error = error
                self.queue.close(f"the network server failed: {type(error).__name__}: {error}")
                raise

    @torch.no_grad()
    def answer(self, batch: dict):
        """Runs every network on its rows of ``batch`` (an ``ExternalQueue.take`` dict) and answers them."""
        models = batch['model']
        count = len(models)
        logits = np.empty((count, self.queue.logits_size), dtype=np.float32)
        values = np.empty(count, dtype=np.float32)
        for model in np.unique(models):
            rows = np.nonzero(models == model)[0]
            network = self.models[int(model)]
            obs = {key: torch.from_numpy(value[rows]).to(self.device, non_blocking=True)
                   for key, value in batch['obs'].items()}
            state = self._state_rows(network, int(model), batch, rows)
            heads, new_state = network.heads(obs, state)
            logits[rows] = heads.flat().float().cpu().numpy()
            values[rows] = heads.value.float().cpu().numpy()
            if new_state is not None:
                self._keep_states(int(model), batch, rows, new_state)
        self.queue.answer(batch['slot'], logits, values)
        self.batches += 1
        self.rows += count

    def _state_rows(self, network, model: int, batch: dict, rows) -> Optional[torch.Tensor]:
        fresh = network.initial_state(len(rows)) if hasattr(network, 'initial_state') else None
        if fresh is None:
            return None
        fresh = fresh.to(self.device)
        for i, row in enumerate(rows):
            kept = self._states.get((model, int(batch['slot'][row])))
            step, job = int(batch['step'][row]), int(batch['job'][row])
            if kept is not None and (batch['bootstrap'][row] or (step > 0 and kept[0] == job)):
                fresh[i] = kept[1]
        return fresh

    def _keep_states(self, model: int, batch: dict, rows, state: torch.Tensor):
        for i, row in enumerate(rows):
            if not batch['bootstrap'][row]:
                self._states[(model, int(batch['slot'][row]))] = (int(batch['job'][row]), state[i].detach())

    def close(self, reason: str = "the network server stopped"):
        """Fails every waiting and later request and ends the thread."""
        self.queue.close(reason)
        if self._thread is not None:
            self._thread.join()


_DEFAULT: Optional[Server] = None


def default_server() -> Server:
    """The process's server (created on first use)."""
    global _DEFAULT  # pylint: disable=global-statement
    if _DEFAULT is None or _DEFAULT.queue.closed:
        _DEFAULT = Server()
    return _DEFAULT
