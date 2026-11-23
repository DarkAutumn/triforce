# Back-test trees: every version's code with its team (the 250-run test on old versions)

The owner (Nov 23): run the 250-run test on every version that can run, in trees left usable. One worktree per
version with a complete team in `/backup/triforce/INDEX.md` (every network present by checkpoint SHA-1):

- `~/git/triforce-eval250-v<version>`, branch `eval250/v<version>` at the version's tag (untagged versions: the
  commit their team's chains played: 1.4.2.0 `a75e40c`, 1.5.1.0 `415e030`, 2.0.0.4 `cd1499ad`); no fixups were
  needed, so every branch is still at that commit.
- `.venv` (Python 3.12, `uv pip install -e '.[dev]'`, which builds the version's own `triforce._native` in release)
  and `roms -> ~/git/triforce/roms`.
- `models/`: `scripts/backup_models.py team <version> models --route <route> --link` (branch r39-backup): each run
  dir a symlink into `/backup/triforce/models/runs/` (read-only files), the pointers `promoted` /
  `promoted.<route>` written so the version's registry resolves every network of `models/teams/<version>.json`.
  Play with `--models-dir models` (a global option, before the subcommand), e.g.
  `.venv/bin/python -m triforce.cli --models-dir models eval-chain --route <route> --runs 250 ...`; the versions
  before `scripts/full_runs.py` exists have no power-on 250 driver of their own.
- The smoke: `eval-chain --route <route> --runs 2 --parallel 2 --first-seed 0 --stop-at step:20` through the job
  line, output in `target/eval250-smoke/` (`smoke.jsonl`, its `.meta.json` provenance, `smoke.log`, `rc`);
  "played N of the manifest's N networks by SHA-1" compares that provenance with the manifest.
- Route: the version's route of record (`d3first` from 2.0.0.0, `d2first` before). The 2.0.x versions' `d2first`
  teams are in the manifests too (`team <version> DIR --route d2first`), not built here.

This table is rewritten after each smoke finishes (they queue one at a time behind the machine's other jobs).

| version | worktree | branch @ commit | models dir (route) | builds and loads | smoke (2 chain runs to step 20, queued) | missing / notes |
|---|---|---|---|---|---|---|
| 1.3.1.0 | `/home/leculver/git/triforce-eval250-v1.3.1.0` | `eval250/v1.3.1.0` @ 5a5d4d23 (v1.3.1.0) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 20 of the manifest's 20 networks by SHA-1 | - |
| 1.3.2.0 | `/home/leculver/git/triforce-eval250-v1.3.2.0` | `eval250/v1.3.2.0` @ 8ee1fc23 (v1.3.2.0) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=2: no records | triforce eval-chain: error: models/boss-aquamentus/r66-dg/ckpt.pt: model version 1.4.0.0 is not compatible with triforce 1.3.2.0 (the first three numbers must match); stamp a model known to be compati |
| 1.4.0.0 | `/home/leculver/git/triforce-eval250-v1.4.0.0` | `eval250/v1.4.0.0` @ 04b18b83 (v1.4.0.0) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 24 of the manifest's 24 networks by SHA-1 | - |
| 1.4.1.0 | `/home/leculver/git/triforce-eval250-v1.4.1.0` | `eval250/v1.4.1.0` @ 74109a5e (v1.4.1.0) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=1: no records | ValueError: models/boss-aquamentus/r71-e/act: act mode 'temperature:0.5+relax:16' is not one of ('sample', 'argmax') |
| 1.4.1.1 | `/home/leculver/git/triforce-eval250-v1.4.1.1` | `eval250/v1.4.1.1` @ 637fdde6 (v1.4.1.1) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=1: no records | ValueError: models/boss-aquamentus/r71-e/act: act mode 'temperature:0.5+relax:16' is not one of ('sample', 'argmax') |
| 1.4.1.2 | `/home/leculver/git/triforce-eval250-v1.4.1.2` | `eval250/v1.4.1.2` @ 9478d1e5 (v1.4.1.2) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 24 of the manifest's 24 networks by SHA-1 | - |
| 1.4.2.0 | `/home/leculver/git/triforce-eval250-v1.4.2.0` | `eval250/v1.4.2.0` @ a75e40c6 (a75e40c) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=2: no records | triforce eval-chain: error: models/boss-aquamentus/r71-e/ckpt.pt: model version 1.5.0.0 is not compatible with triforce 1.4.2.0 (the first three numbers must match); stamp a model known to be compatib |
| 1.5.0.0 | `/home/leculver/git/triforce-eval250-v1.5.0.0` | `eval250/v1.5.0.0` @ 2b297cb6 (v1.5.0.0) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 failure, seed 1 success; played 28 of the manifest's 28 networks by SHA-1 | - |
| 1.5.0.1 | `/home/leculver/git/triforce-eval250-v1.5.0.1` | `eval250/v1.5.0.1` @ b6d4e351 (v1.5.0.1) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 28 of the manifest's 28 networks by SHA-1 | - |
| 1.5.1.0 | `/home/leculver/git/triforce-eval250-v1.5.1.0` | `eval250/v1.5.1.0` @ 415e0307 (415e030) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=2: no records | triforce eval-chain: error: models/boss-aquamentus/r76-b/ckpt.pt: model version 2.0.0.0 is not compatible with triforce 1.5.1.0 (the first three numbers must match); stamp a model known to be compatib |
| 1.6.0.0 | `/home/leculver/git/triforce-eval250-v1.6.0.0` | `eval250/v1.6.0.0` @ 6b4157f5 (v1.6.0.0) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=2: no records | triforce eval-chain: error: models/boss-aquamentus/r76-b/ckpt.pt: model version 2.0.0.0 is not compatible with triforce 1.6.0.0 (the first three numbers must match); stamp a model known to be compatib |
| 1.7.0.0 | `/home/leculver/git/triforce-eval250-v1.7.0.0` | `eval250/v1.7.0.0` @ e7065bae (v1.7.0.0) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=2: no records | triforce eval-chain: error: models/boss-aquamentus/r76-b/ckpt.pt: model version 2.0.0.0 is not compatible with triforce 1.7.0.0 (the first three numbers must match); stamp a model known to be compatib |
| 1.9.0.0 | `/home/leculver/git/triforce-eval250-v1.9.0.0` | `eval250/v1.9.0.0` @ 10d4522a (v1.9.0.0) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=2: no records | triforce eval-chain: error: models/boss-aquamentus/r76-b/ckpt.pt: model version 2.0.0.0 is not compatible with triforce 1.9.0.0 (the first three numbers must match); stamp a model known to be compatib |
| 1.9.0.1 | `/home/leculver/git/triforce-eval250-v1.9.0.1` | `eval250/v1.9.0.1` @ 8b48df60 (v1.9.0.1) | `models` (d2first) | yes (venv, release `_native`, `import triforce._native`) | FAILED rc=2: no records | triforce eval-chain: error: models/boss-aquamentus/r76-b/ckpt.pt: model version 2.0.0.0 is not compatible with triforce 1.9.0.1 (the first three numbers must match); stamp a model known to be compatib |
| 2.0.0.0 | `/home/leculver/git/triforce-eval250-v2.0.0.0` | `eval250/v2.0.0.0` @ 9ee2f6a5 (v2.0.0.0) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 failure, seed 1 success; played 49 of the manifest's 49 networks by SHA-1 | - |
| 2.0.0.1 | `/home/leculver/git/triforce-eval250-v2.0.0.1` | `eval250/v2.0.0.1` @ 273a1ed5 (v2.0.0.1) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 failure; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.2 | `/home/leculver/git/triforce-eval250-v2.0.0.2` | `eval250/v2.0.0.2` @ 83d9960c (v2.0.0.2) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.3 | `/home/leculver/git/triforce-eval250-v2.0.0.3` | `eval250/v2.0.0.3` @ 7687066f (v2.0.0.3) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.4 | `/home/leculver/git/triforce-eval250-v2.0.0.4` | `eval250/v2.0.0.4` @ cd1499ad (cd1499ad) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.5 | `/home/leculver/git/triforce-eval250-v2.0.0.5` | `eval250/v2.0.0.5` @ cde00236 (v2.0.0.5) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.6 | `/home/leculver/git/triforce-eval250-v2.0.0.6` | `eval250/v2.0.0.6` @ cbc1dfa9 (v2.0.0.6) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 failure, seed 1 success; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.7 | `/home/leculver/git/triforce-eval250-v2.0.0.7` | `eval250/v2.0.0.7` @ 1caa6475 (v2.0.0.7) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 failure, seed 1 success; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.8 | `/home/leculver/git/triforce-eval250-v2.0.0.8` | `eval250/v2.0.0.8` @ d8596127 (v2.0.0.8) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.9 | `/home/leculver/git/triforce-eval250-v2.0.0.9` | `eval250/v2.0.0.9` @ 1bf27323 (v2.0.0.9) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 51 of the manifest's 51 networks by SHA-1 | - |
| 2.0.0.10 | `/home/leculver/git/triforce-eval250-v2.0.0.10` | `eval250/v2.0.0.10` @ f1d67ff5 (v2.0.0.10) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 53 of the manifest's 53 networks by SHA-1 | - |
| 2.0.0.11 | `/home/leculver/git/triforce-eval250-v2.0.0.11` | `eval250/v2.0.0.11` @ 0a3e7915 (v2.0.0.11) | `models` (d3first) | yes (venv, release `_native`, `import triforce._native`) | ok: seed 0 success, seed 1 success; played 54 of the manifest's 54 networks by SHA-1 | - |
