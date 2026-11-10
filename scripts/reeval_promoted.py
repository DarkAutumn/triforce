#!/usr/bin/env python3
"""Re-evaluate every promoted specialist model and replace its run's evaluation with a ``format: 2`` one.

    python scripts/reeval_promoted.py [--specialists a,b] [--run-id ID] [--parallel 8] [--episodes-per-state 100]
        [--models-dir models] [--output-dir DIR]

For each specialist (``specialists/*.yaml``, or ``--specialists``) with a promoted model
(``<models>/<name>/promoted``), runs ``triforce.evaluate.evaluate_specialist`` on the promoted checkpoint in the
run's act mode and writes the result atomically to ``triforce.evaluate.default_output``'s file in the run dir:
``eval.json`` with the default 100 episodes per state, ``eval-e<N>.json`` otherwise.  ``--run-id`` evaluates that
run instead of the promoted one (every listed specialist that has it; e.g. a round trained without ``--promote``).
An existing file of that name without ``format: 2`` is first copied to ``eval-format1.json`` beside it.
``--output-dir`` writes ``<output-dir>/<name>/<run-id>/<file>`` instead and leaves the models directory untouched.
Prints one line per specialist; exits 1 if any evaluation failed.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from triforce import registry
from triforce.evaluate import FORMAT, default_output, evaluate_specialist, write_result
from triforce.specialist import list_specialists


def _format(path: Path):
    try:
        with open(path, encoding='utf-8') as file:
            return json.load(file).get('format')
    except (OSError, ValueError, AttributeError):
        return None


def _write(result: dict, output: Path):
    """Writes the evaluation atomically, keeping a format-1 original as eval-format1.json."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and _format(output) != FORMAT:
        shutil.copy2(output, output.with_name('eval-format1.json'))
    write_result(result, output)


def main(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--specialists', default=None, help="comma-separated names (default: every promoted one)")
    parser.add_argument('--parallel', type=int, default=8, help="envs in each evaluation pool")
    parser.add_argument('--episodes-per-state', type=int, default=100)
    parser.add_argument('--models-dir', default=None, help=f"default ${registry.MODELS_DIR_ENV} or <repo>/models")
    parser.add_argument('--run-id', default=None, help="evaluate this run instead of the promoted one")
    parser.add_argument('--output-dir', default=None,
                        help="write <dir>/<name>/<run-id>/eval.json instead of into the models directory")
    args = parser.parse_args(argv)

    names = args.specialists.split(',') if args.specialists else list_specialists()
    failed = 0
    for name in names:
        root = registry.models_root(args.models_dir)
        try:
            run_id = args.run_id or registry.promoted_run(name, args.models_dir)
        except FileNotFoundError:
            if args.specialists:
                print(f"{name}: no promoted model", flush=True)
                failed += 1
            continue
        if not (root / name / run_id / 'ckpt.pt').is_file():
            if args.specialists:
                print(f"{name}: no {run_id} checkpoint", flush=True)
                failed += 1
            continue
        try:
            checkpoint = root / name / run_id
            start = time.monotonic()
            result = evaluate_specialist(name, checkpoint, args.episodes_per_state, args.parallel)
            out_root = Path(args.output_dir) if args.output_dir else root
            name_of_file = default_output(checkpoint, name, 'model', args.episodes_per_state, 0).name
            output = out_root / name / run_id / name_of_file
            _write(result, output)
        except Exception as error:      # pylint: disable=broad-exception-caught
            print(f"{name} {run_id}: FAILED: {error!r}", flush=True)
            failed += 1
            continue
        print(f"{name} {run_id}: success {result['success_rate']:.1%} over {result['episodes']} episodes, "
              f"health change {result['mean_health_change']:+.2f}, {len(result['visited_rooms'])} rooms visited, "
              f"{time.monotonic() - start:.0f}s -> {output}", flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
