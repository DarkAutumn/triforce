"""``triforce store``: where the shared things are and what the shared store holds (``triforce.store``).

``paths`` prints every location as JSON; ``data``, ``runs``, ``teams`` list what is there; ``summaries`` the kept
rows of a run's or a team's measurements over time; ``outputs`` what a job filed; ``log`` the audit log (every
write: when, which job and agent, what); ``view TEAM --into DIR`` writes an old-style models folder of links for an
old version's tree.  ``checkpoint FAMILY/RUN@itN`` (round 41) is the one command here that adds to the store: a
numbered checkpoint of a run, open or sealed, made a sealed run of its own (``store.checkpoint_run``: only added,
never changing the parent; ``--plan`` and ``--list`` only read).  Nothing else here writes the store but ``view``
(which writes only DIR) and ``reindex`` (derived tables).
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


def cmd_checkpoint(args) -> int:
    text, iteration = store.split_checkpoint(args.run)
    if iteration is not None and args.it is not None:
        raise store.StoreError(f"{args.run}: the checkpoint is named twice (@ and --it): give one")
    parent = store.parse_run(text)
    if args.list:
        row = store.get_run(parent)
        found = store.numbered_checkpoints(parent, row['status'] == store.SEALED)
        if args.json:
            _print_json([{**c, 'dir': str(c['dir'])} for c in found])
            return 0
        print(f"{parent.identity} ({row['status']}): {len(found)} numbered checkpoints")
        for c in found:
            made = store.run_row(store.checkpoint_run_name(parent, c['iteration']))
            print(f"  it{c['iteration']:06d}  " + (f"complete {c['sha1'][:12]}" if c['complete'] else
                                                   f"being written: {c['why']}")
                  + (f"  -> {store.checkpoint_run_name(parent, c['iteration']).identity}" if made else ''))
        return 0
    wanted = iteration if iteration is not None else (args.it or 'latest')
    if wanted != 'latest' and not str(wanted).removeprefix('it').isdigit():
        raise store.StoreError(f"--it {wanted}: an iteration (228, it228) or latest")
    wanted = wanted if wanted == 'latest' else int(str(wanted).removeprefix('it'))
    result = store.checkpoint_run(parent, wanted, make=not args.plan)
    if args.json:
        _print_json(result)
        return 0
    state = ('exists' if result['exists'] else 'made now' if result['made'] else
             f"would be made ({result['weights']} from the {result['parent_status']} run)")
    steps = f", {result['steps']:,} steps" if result.get('steps') is not None else ''
    print(f"{result['run']}: iteration {result['iteration']} of {result['parent']}{steps}, sha1 "
          f"{result['checkpoint_sha1'][:12]}, {result['weights']}; {state}")
    return 0


def cmd_query(args) -> int:
    from .store_stats import EXAMPLES  # pylint: disable=import-outside-toplevel
    if args.sql is None:
        for name, (what, sql) in EXAMPLES.items():
            print(f"{name}: {what}\n  {sql}\n")
        return 0
    sql = EXAMPLES[args.sql][1] if args.sql in EXAMPLES else args.sql
    params = {}
    for item in args.param:
        key, _, value = item.partition('=')
        try:
            params[key] = int(value, 0)         # 3, 0x5b
        except ValueError:
            params[key] = value
    columns, rows = store.query(sql, params)
    if args.json:
        _print_json([dict(zip(columns, row)) for row in rows])
        return 0
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(str(c))
              for i, c in enumerate(columns)]
    print('  '.join(str(c).ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        print('  '.join(str(v).ljust(w) for v, w in zip(row, widths)))
    return 0


def cmd_reindex(args) -> int:  # pylint: disable=unused-argument
    print(json.dumps(store.reindex(), indent=2))
    return 0


def main(argv=None) -> int:
    """``triforce store ...``."""
    parser = argparse.ArgumentParser(prog='triforce store', description="Where the shared things are and what the "
                                     "shared store holds (read-only but `checkpoint`, which only adds a sealed run: "
                                     "tools write the store through triforce.store).")
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
    checkpoint = sub.add_parser('checkpoint', help="a run's numbered checkpoint as a sealed run a team can play, "
                                                   "FAMILY/RUN-it<NNNNNN> (made once; the parent only read)")
    checkpoint.add_argument('run', metavar='FAMILY/RUN[@itN|@latest]')
    checkpoint.add_argument('--it', default=None, metavar='N|latest',
                            help="the iteration (default latest: the newest completely written)")
    checkpoint.add_argument('--plan', action='store_true', help="say which run it is and whether it exists; make "
                                                                "nothing")
    checkpoint.add_argument('--list', action='store_true', help="the run's numbered checkpoints, complete or not")
    query = sub.add_parser('query', help="the statistics tables (evaluations, evaluation_rooms) in SQL: an example's "
                                         "name or a SELECT; without one, the examples")
    query.add_argument('sql', nargs='?')
    query.add_argument('--param', action='append', default=[], metavar='NAME=VALUE',
                       help="a :NAME parameter (deaths-by-room-over-time: --param level=3 --param room=0x5b)")
    sub.add_parser('reindex', help="rebuild the statistics tables from the filed statistics files")
    for parser_ in (data, runs, teams, summaries, outputs, log, query, checkpoint):
        parser_.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    commands = {'paths': cmd_paths, 'data': cmd_data, 'runs': cmd_runs, 'teams': cmd_teams,
                'summaries': cmd_summaries, 'outputs': cmd_outputs, 'log': cmd_log, 'view': cmd_view,
                'query': cmd_query, 'reindex': cmd_reindex, 'checkpoint': cmd_checkpoint}
    try:
        return commands[args.command](args)
    except (store.StoreError, paths.MissingError) as error:
        print(f"triforce store: error: {error}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
