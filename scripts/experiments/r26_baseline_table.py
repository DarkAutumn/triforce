"""Round 26: the per-room table of a chain measurement of the leg to Dungeon 7 and of Dungeon 7, in Markdown
(docs/experiments/round26.md "The baseline").

    python scripts/experiments/r26_baseline_table.py ROOM_STATS.json [SCOPE ...]

ROOM_STATS.json is what ``triforce room-stats TAG`` writes; SCOPE defaults to ``to-d7 d7``.  One row per room and
route step of the scope that a run entered (visits off the route as ``off``): visits, deaths, wrong exits, timeouts
(the outcome ``max-steps``), stuck endings, every other outcome that is no correct exit, hearts lost a visit and the
cheat ledger's bombs, keys and rupees.
"""

import json
import sys

KNOWN = ('exit-correct', 'triforce-collected', 'death', 'exit-wrong', 'max-steps', 'stuck')


def rows(scope: dict):
    for key, room in scope['rooms'].items():
        for step, entry in room['steps'].items():
            yield (int(step) if step.isdigit() else 10 ** 6, step, key, entry)


def table(scope: dict) -> str:
    endings = ', '.join(f'{name} {count}' for name, count in sorted(scope['endings'].items()))
    lines = [f"{scope['label']}: {scope['runs']} runs, arrived {scope['arrived']}; runs ended: {endings}", '',
             '| step | room | family | visits | correct | deaths | wrong exits | timeouts | stuck | other | hearts lost a visit '
             '| cheated bombs | keys | rupees |', '|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for _, step, key, entry in sorted(rows(scope), key=lambda row: row[0]):
        outcomes = entry['outcomes']
        other = ', '.join(f'{name} {count}' for name, count in sorted(outcomes.items()) if name not in KNOWN)
        correct = outcomes.get('exit-correct', 0) + outcomes.get('triforce-collected', 0)
        cheated = entry['cheated']
        family = ', '.join(sorted(entry['specialists']))
        lines.append(f"| {step} | {key} | {family} | {entry['visits']} | {correct} | {entry['deaths']} | "
                     f"{outcomes.get('exit-wrong', 0)} | {outcomes.get('max-steps', 0)} | {outcomes.get('stuck', 0)} | "
                     f"{other} | {entry['hearts_lost_per_visit']:.2f} | {cheated['bombs']} | {cheated['keys']} | "
                     f"{cheated['rupees']} |")
    return '\n'.join(lines)


def main():
    data = json.load(open(sys.argv[1], encoding='utf-8'))
    for name in sys.argv[2:] or ('to-d7', 'd7'):
        if name in data['scopes']:
            print(table(data['scopes'][name]), end='\n\n')


if __name__ == '__main__':
    main()
