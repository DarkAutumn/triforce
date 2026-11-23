"""``triforce store``: where the shared things are and what the shared store holds (``triforce.store``).

``paths`` prints every location as JSON; ``data``, ``runs``, ``teams`` list what is there; ``summaries`` the kept
rows of a run's or a team's measurements over time; ``outputs`` what a job filed; ``log`` the audit log (every
write: when, which job and agent, what); ``view TEAM --into DIR`` writes an old-style models folder of links for an
old version's tree.  Nothing here writes the store but ``view`` (which writes only DIR).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from . import paths, store


def _since(text: str) -> float:
    units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    if text and text[-1] in units:
        return time.time() - float(text[:-1]) * units[text[-1]]
    return time.time() - float(text)


def _when(ts) -> str:
    return '-' if ts is None else time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))


def _print_json(value):
    print(json.dumps(value, indent=2, default=str))


def cmd_paths(args) -> int:
    _print_json(paths.layout())
    return 0


def cmd_data(args) -> int:
    rows = store.snapshots(args.kind)
    if args.json:
        _print_json(rows)
        return 0
    for row in rows:
        print(f"{row['kind']:16} {row['name']:24} {'current' if row['current'] else '':8} {row['files']:8} files "
              f"{row['bytes'] / 1e6:10.1f} MB  sha256 {row['sha256'][:16]}  {row['source']}")
    return 0


def cmd_runs(args) -> int:
    if args.run and '/' in args.run and store.run_row(store.parse_run(args.run)) is not None:
        row = store.get_run(store.parse_run(args.run))
        row['evaluations'] = [o['path'] for o in store.outputs(run=args.run)]
        _print_json(row)
        return 0
    rows = store.runs(args.run, store.OPEN if args.open else None)
    if args.json:
        _print_json(rows)
        return 0
    for row in rows:
        print(f"{row['family'] + '/' + row['name']:60} {row['status']:6} {(row['sha1'] or '-')[:12]:12} "
              f"{row['act'] or '-':26} {row['origin'] or '-':10} {row['parent'] or '':30} "
              f"{row['owner_agent'] or ''} {_when(row['sealed'] or row['created'])}")
    return 0


def cmd_teams(args) -> int:
    if args.name:
        team = store.effective_team(args.name) if '+' in args.name else store.read_team(args.name)
        if args.json:
            _print_json(team)
            return 0
        print(f"team {team['name']} (sha256 {team['sha256'][:16]})")
        for key, slot in team['slots'].items():
            print(f"  {key:32} {slot['run']:48} {slot['sha1'][:12]} {slot['act']}"
                  f"{'  hit head' if slot.get('hit_head') else ''}")
        return 0
    rows = store.teams()
    if args.json:
        _print_json(rows)
        return 0
    for row in rows:
        print(f"{row['name']:16} {'current' if row['current'] else '':8} {_when(row['created'])}  {row['source']}")
    return 0


def cmd_summaries(args) -> int:
    rows = store.summaries(args.run, args.team, args.kind)
    if args.json:
        _print_json(rows)
        return 0
    for row in rows:
        rate = '-' if row['rate'] is None else f"{row['rate']:.1%} ± {row['se'] or 0:.1%}"
        print(f"{_when(row['ts'])} {row['kind']:9} {row['name'][:48]:48} n={row['n'] or '-':>6} {rate:16} "
              f"job {row['job'] or '-'} {row['run'] or row['team'] or ''}")
    return 0


def cmd_outputs(args) -> int:
    rows = store.outputs(args.job, args.run, args.kind)
    if args.json:
        _print_json(rows)
        return 0
    for row in rows:
        print(f"{_when(row['created'])} job {row['job'] or '-':10} {row['kind']:10} {row['path']}"
              f"{'  run ' + row['run'] if row['run'] else ''}{'  team ' + row['team'] if row['team'] else ''}")
    return 0


def cmd_log(args) -> int:
    rows = store.audit_log(_since(args.since) if args.since else None, args.job, args.last)
    if args.json:
        _print_json(rows)
        return 0
    for row in rows:
        print(f"{_when(row['ts'])} job {row['job'] or '-':8} {row['agent'] or row['user'] or '-':14} "
              f"{row['entry']:6} {row['op']:18} {json.dumps(row['what'], sort_keys=True)}")
    return 0


def cmd_view(args) -> int:
    team = store.effective_team(args.team)
    store.write_view(team, args.into)
    print(f"{args.into}: team {team['name']}, {len(team['slots'])} slots (read-only links into {store.root()})")
    return 0


def main(argv=None) -> int:
    """``triforce store ...``."""
    parser = argparse.ArgumentParser(prog='triforce store', description="Where the shared things are and what the "
                                     "shared store holds (read-only: tools write it through triforce.store).")
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('paths', help="every location as JSON (the store's layout.json, this checkout's scratch)")
    data = sub.add_parser('data', help="the named data snapshots")
    data.add_argument('kind', nargs='?', choices=paths.DATA_KINDS)
    runs = sub.add_parser('runs', help="the network runs (FAMILY to list one family's, FAMILY/RUN for one run)")
    runs.add_argument('run', nargs='?', metavar='FAMILY[/RUN]')
    runs.add_argument('--open', action='store_true', help="only the open runs (trainings writing, or interrupted)")
    teams = sub.add_parser('teams', help="the teams; NAME (or NAME+SLOT=RUN...) shows one's slots")
    teams.add_argument('name', nargs='?')
    summaries = sub.add_parser('summaries', help="the kept summary rows of a run's or a team's measurements")
    summaries.add_argument('--run', metavar='FAMILY/RUN')
    summaries.add_argument('--team')
    summaries.add_argument('--kind', choices=('chain', 'evaluate', 'train'))
    outputs = sub.add_parser('outputs', help="what jobs filed (evaluations, chain records, replay packs, ...)")
    outputs.add_argument('--job')
    outputs.add_argument('--run', metavar='FAMILY/RUN')
    outputs.add_argument('--kind')
    log = sub.add_parser('log', help="the audit log: every write of the store")
    log.add_argument('--since', help="e.g. 24h, 30m, 7d")
    log.add_argument('--job')
    log.add_argument('--last', type=int)
    view = sub.add_parser('view', help="an old-style models folder of links for TEAM (an old version's tree)")
    view.add_argument('team', help="NAME, current, or NAME+SLOT=RUN...")
    view.add_argument('--into', type=Path, required=True)
    for parser_ in (data, runs, teams, summaries, outputs, log):
        parser_.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    commands = {'paths': cmd_paths, 'data': cmd_data, 'runs': cmd_runs, 'teams': cmd_teams,
                'summaries': cmd_summaries, 'outputs': cmd_outputs, 'log': cmd_log, 'view': cmd_view}
    try:
        return commands[args.command](args)
    except (store.StoreError, paths.MissingError) as error:
        print(f"triforce store: error: {error}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
