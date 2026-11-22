"""Rehearsal: teacher-labelled rows of tasks a lab run does not roll out, so a network trained on some families keeps
playing the others (``python -m triforce.lab.train --rehearse CACHE``; docs/lab.md "Rehearsal").

The rows are a distillation cache (``scripts/experiments/r36_merge_distill.py cache``: one ``.npy`` a field, the
teachers' joint log-probs as float16, each row's key in ``key.npy``, the key names in ``cache.json``).  ``rows`` of
them, the same number from every key named, are read into host memory once; every PPO minibatch adds
``coeff`` · KL(teacher ‖ π_θ) on a sample of them as large as the minibatch (``ppo.PPO.rehearsal``)."""
import json
from pathlib import Path

import numpy as np
import torch

LONG_FIELDS = ('room_id', 'entity_types', 'grid')


class Rehearsal:
    """A fixed sample of a cache's rows for ``keys``, drawn per minibatch."""

    def __init__(self, cache: Path, keys: list[str], rows: int, seed: int = 0):
        cache = Path(cache)
        meta = json.loads((cache / 'cache.json').read_text(encoding='utf-8'))
        names = meta['keys']
        missing = [k for k in keys if k not in names]
        if missing:
            raise SystemExit(f"--rehearse {cache}: no rows of {missing}")
        key_of = np.load(cache / 'key.npy')
        rng = np.random.default_rng(seed)
        per_key = max(1, rows // max(1, len(keys)))
        picked = []
        for key in keys:
            where = np.flatnonzero(key_of == names.index(key))
            picked.append(rng.choice(where, size=min(per_key, len(where)), replace=False))
        index = np.sort(np.concatenate(picked))
        self.keys = list(keys)
        self.obs = {}
        for field in meta['fields']:
            array = np.load(cache / f'{field.replace("/", "_")}.npy', mmap_mode='r')
            values = torch.from_numpy(np.ascontiguousarray(array[index]))
            if field.startswith('obs/'):
                name = field[4:]
                self.obs[name] = values.long() if name in LONG_FIELDS else values
            elif field == 'action_mask':
                self.masks = values
            elif field == 'teacher_joint':
                self.joint = values
        self.rows = len(index)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, n: int, device) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """(observations, masks, teacher joint log-probs as float32) of ``n`` rows drawn with replacement."""
        pick = torch.randint(self.rows, (n,), generator=self.generator)
        obs = {k: v[pick].to(device, non_blocking=True) for k, v in self.obs.items()}
        return obs, self.masks[pick].to(device), self.joint[pick].to(device).float()
