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
import weakref
from typing import Optional

import numpy as np
import torch

from .. import _native

DEFAULT_GATHER = 0.005      # at most this many seconds a batch waits for the other active envs' rows (eager forwards)
GRAPH_GATHER = 0.001        # ... with CUDA graphs (a forward is cheap: answering sooner beats a fuller batch)
ACTIVE_SECONDS = 1.0        # an env that asked this recently is waited for (``Server._loop``)
IDLE_SECONDS = 3600.0       # with no env active the thread blocks this long for a request (``close`` wakes it)
CPU_THREADS = 4             # torch threads when serving on the CPU
GRAPH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)    # batch sizes a CUDA graph is captured for; larger batches run eagerly
GRAPH_CHECKS = 3            # a new graph's first replays are compared with the eager forward on the same rows
GRAPH_TOLERANCE = 1e-4      # ... and must agree within this (a graph replays the captured kernels; 0 measured)


def serve_device() -> torch.device:
    """``$TRIFORCE_SERVE_DEVICE`` (cpu | cuda), else cuda when available."""
    name = os.environ.get('TRIFORCE_SERVE_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def serve_half() -> bool:
    """``$TRIFORCE_SERVE_HALF=1``: run served networks in half precision on the GPU (autocast float16; measured in
    docs/lab.md "Acting speed"; the logits differ from float32 within the half's precision)."""
    return os.environ.get('TRIFORCE_SERVE_HALF', '') not in ('', '0')


_LEARNER = {'on': False}


def learner_process():
    """Marks this process as one whose own threads use the GPU while the server serves (a training, a recording):
    its servers run eagerly.  A CUDA graph capture on the server thread cannot coexist with another thread's GPU
    work in the same process (Nov 23: a training's value recompute and the capture failed each other with
    cudaErrorStreamCaptureUnsupported, in every capture mode)."""
    _LEARNER['on'] = True


def serve_graphs() -> bool:
    """CUDA graphs for served networks: on by default on the GPU, off in a learner process (``learner_process``) and
    with ``$TRIFORCE_SERVE_GRAPHS=0``."""
    return not _LEARNER['on'] and os.environ.get('TRIFORCE_SERVE_GRAPHS', '1') not in ('', '0')


def _bucket(rows: int) -> Optional[int]:
    return next((b for b in GRAPH_BUCKETS if b >= rows), None)


class _Graph:
    """One network's forward at one batch size, captured as a CUDA graph: static inputs filled in place, one
    replay, the logits and value side by side in a static output. A replay launches the captured kernels in one
    call, so it costs about a millisecond where the eager forward's Python and launches cost 4-5 (docs/lab.md
    "Acting speed")."""

    def __init__(self, network, obs: dict, half: bool):
        self.network, self.half = network, half
        self.inputs = {key: value.clone() for key, value in obs.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):      # warm-up off the default stream (lazy initialisation, cuBLAS workspaces)
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._forward()
        self.checks = GRAPH_CHECKS

    def _forward(self) -> torch.Tensor:
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.half):
            heads, _ = self.network.heads(self.inputs)
        return _flat(heads)

    def run(self, obs: dict) -> torch.Tensor:
        for key, value in obs.items():
            self.inputs[key].copy_(value, non_blocking=True)
        self.graph.replay()
        return self.output


def _flat(heads) -> torch.Tensor:
    """The logits and the value side by side (one copy back)."""
    return torch.cat([heads.flat().float(), heads.value.float().unsqueeze(-1)], dim=-1)


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

    def __init__(self, device=None, gather: Optional[float] = None, threads: int = CPU_THREADS,
                 half: Optional[bool] = None, graphs: Optional[bool] = None):
        self.device = torch.device(device) if device is not None else serve_device()
        self.half = (serve_half() if half is None else half) and self.device.type == 'cuda'
        # CUDA graphs (``_Graph``) for networks without recurrent state, per batch size bucket; a network whose
        # capture fails or whose replay disagrees with its eager forward is served eagerly from then on.
        self.graphs = (serve_graphs() if graphs is None else graphs) and self.device.type == 'cuda'
        default = GRAPH_GATHER if self.graphs else DEFAULT_GATHER
        self.gather = float(os.environ.get('TRIFORCE_SERVE_GATHER', default)) if gather is None else gather
        self._graphs: dict[tuple[int, int], _Graph] = {}
        self._eager: set[int] = set()
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
        ``key`` (default: the network object) reuses its model index and replaces the network there; a new key
        always gets a new index."""
        key = id(network) if key is None else key
        with self._lock:
            index = self._keys.get(key)
            if index is None:
                index = self._keys[key] = len(self.models)
            self.models[index] = network.to(self.device).eval()
            self._states = {k: v for k, v in self._states.items() if k[0] != index}
            self._graphs = {k: v for k, v in self._graphs.items() if k[0] != index}
            self._eager.discard(index)
        self._start()
        return self.queue.policy(index, weights_hash(network))

    def serve_copy(self, network: torch.nn.Module, key=None):
        """``policy`` of a frozen copy of ``network`` (``key`` default: the network object, so a training run's
        rollouts reuse one index and the copy before is dropped).  The default key lives as long as ``network``:
        once it is gone its id may name another network (``load_model(path)[0].to_native()`` in a loop), which
        must get its own index, never replace the one served for the network that died."""
        if key is None:
            key = id(network)
            if key not in self._keys:
                weakref.finalize(network, self._keys.pop, key, None)
        return self.policy(copy.deepcopy(network), key)

    def _start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name='net-server', daemon=True)
            self._thread.start()

    def _loop(self):
        while not self.queue.closed:
            # Wait for every env that asked in the last second (they step their emulators at different times; a
            # batch per straggler costs a forward each), at most ``gather`` seconds.  With none active the thread
            # blocks until a request comes (Nov 23: an idle server woke ten times a second, per served network).
            now = time.monotonic()
            self._seen = {slot: t for slot, t in self._seen.items() if now - t < ACTIVE_SECONDS}
            batch = self.queue.take(timeout=0.1 if self._seen else IDLE_SECONDS, gather=self.gather,
                                    target=max(1, len(self._seen)))
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
            state = self._state_rows(network, int(model), batch, rows)
            out = self._graphed(int(model), network, batch, rows) if state is None and self.graphs else None
            new_state = None
            if out is None:
                obs = {key: torch.from_numpy(value[rows]).to(self.device, non_blocking=True)
                       for key, value in batch['obs'].items()}
                with torch.autocast('cuda', dtype=torch.float16, enabled=self.half):
                    heads, new_state = network.heads(obs, state)
                out = _flat(heads).cpu().numpy()        # one copy back: the logits and the value side by side
            logits[rows] = out[:, :-1]
            values[rows] = out[:, -1]
            if new_state is not None:
                self._keep_states(int(model), batch, rows, new_state)
        self.queue.answer(batch['slot'], logits, values)
        self.batches += 1
        self.rows += count

    def _graphed(self, model: int, network, batch: dict, rows) -> Optional[np.ndarray]:
        """``rows`` of ``batch`` through ``network``'s CUDA graph for their bucket (the rows padded by repeating the
        last; rows are independent), as the logits and value of each row; None when it is served eagerly (a batch
        above the largest bucket, a network marked eager).  A capture that fails, or one of a graph's first replays
        that disagrees with the eager forward on the same rows, marks the network eager with a warning."""
        bucket = _bucket(len(rows))
        if bucket is None or model in self._eager:
            return None
        padded = np.concatenate([rows, np.repeat(rows[-1:], bucket - len(rows))])
        obs = {key: torch.from_numpy(value[padded]).to(self.device, non_blocking=True)
               for key, value in batch['obs'].items()}
        graph = self._graphs.get((model, bucket))
        try:
            if graph is None:
                graph = self._graphs[(model, bucket)] = _Graph(network, obs, self.half)
            out = graph.run(obs)
            if graph.checks:
                graph.checks -= 1
                with torch.autocast('cuda', dtype=torch.float16, enabled=self.half):
                    eager = _flat(network.heads(obs)[0])
                if not torch.allclose(out, eager, atol=GRAPH_TOLERANCE, rtol=GRAPH_TOLERANCE):
                    raise RuntimeError(f"graph and eager forward differ by {float((out - eager).abs().max()):.3g}")
            return out[:len(rows)].cpu().numpy()
        except Exception as error:     # pylint: disable=broad-exception-caught
            self._eager.add(model)
            self._graphs = {k: v for k, v in self._graphs.items() if k[0] != model}
            print(f"triforce: warning: served network {model} runs without CUDA graphs: "
                  f"{type(error).__name__}: {error}", flush=True)
            return None

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
