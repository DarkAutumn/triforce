# ~/triforce-shared: the project's shared store

Everything the project's runs share lives here, outside every checkout: the ROM, the data sets as named snapshots,
every network run, the teams, and what each job filed. **Nobody writes here by hand.** One program does:
`triforce.store` (the tools call it: a training gets its run folder from it and seals it at the end; an evaluation,
a chain, a harvest, a replay pack is filed by the job that made it). What is read-only on disk is meant to stay so:
a write that fails here is an accident the store stopped.

```
README.md, layout.json   this file; every location as JSON (for tools that are not Python)
store.db, locks/         the index (snapshots, runs, teams, jobs, filed outputs, kept summaries) and the audit log
rom/zelda.nes            the ROM (read-only)
data/<kind>/<name>/      named data snapshots, read-only: harvested, harvested-rooms, trajectories, demos;
data/<kind>/current      the name of the kind's default snapshot
runs/<family>/<run>/     network runs: open while their training writes them, then sealed (read-only, checkpoint
                         SHA-1 and default act mode recorded); a continuation, a converted model or a chosen
                         checkpoint is a new run with a parent
teams/<name>.json        teams (immutable): per slot (family, family@route, family@route:late) the run, its
                         checkpoint SHA-1, its act mode and whether it has a hit head
teams/current            the name of the team of record
jobs/<job id>-<label>/   what each job filed: evaluations, chain records, statistics, replay packs, harvests
legacy/<import>/         what the import of the old models folder found no home for (its report lists it)
```

To see it: `triforce store paths` (every location), `triforce store data`, `triforce store runs [FAMILY]`,
`triforce store teams [NAME]`, `triforce store summaries --run FAMILY/RUN | --team NAME`, `triforce store log`
(every write: when, which job and agent, what). A tool plays a team with `--team NAME` (default the current one),
`--team NAME+family@route=RUN` for an experiment; reads a data snapshot by name (`--data KIND=NAME`, default the
kind's current one). An old version's tree that expects a `models/` folder gets one with
`triforce store view TEAM --into DIR` (links into here, read-only).

The archive is `/backup/triforce`; the store's backup copies into it.
