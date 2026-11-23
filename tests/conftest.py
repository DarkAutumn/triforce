"""Shared fixtures on the native stack (``triforce._native``).

Gameplay states come from the corpus base movie: ``pytest_configure`` points ``TRIFORCE_STATES_DIR`` at a fresh
session directory before anything imports ``triforce`` or loads the native game data, and ``pytest_sessionstart``
writes ``start.state`` (the base movie's first gameplay frame, overworld room 0x77) into it, so every default
states dir (the native module's, ``triforce.paths.STATES_DIR``, child processes') is that directory whatever ran
before.  Named route states (``1_44w_0``, ...) live in ``triforce/states``; tests that need one call
``require_state(name)``, which fails (not skips) with the command to build it, and pass its path or
``REPO_STATES_DIR`` explicitly.

This module imports ``triforce`` only inside functions: ``triforce.paths`` reads ``TRIFORCE_STATES_DIR`` at import.
"""

import faulthandler
import json
import os
import shutil
import signal
import tempfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPO_STATES_DIR = ROOT / 'triforce' / 'states'       # the shipped route states
BASE_MOVIE = ROOT / 'corpus' / 'movies' / 'power_on_to_start.tfmovie'
STATES_DIR_ENV = 'TRIFORCE_STATES_DIR'
ROM_MISSING = "ROM not found: set TRIFORCE_ROM or create roms/zelda.nes"
ALLOW_NO_ROM_ENV = 'TRIFORCE_ALLOW_NO_ROM'     # =1: a missing ROM skips ROM tests instead of failing them

_session = {}       # 'states_dir': the session's states dir; 'old_env': TRIFORCE_STATES_DIR before the session


def rom_path() -> Path:
    """$TRIFORCE_ROM, else <repo>/roms/zelda.nes."""
    return Path(os.environ.get('TRIFORCE_ROM', ROOT / 'roms' / 'zelda.nes'))


def rom_missing_allowed() -> bool:
    """True when ``TRIFORCE_ALLOW_NO_ROM=1``: ROM tests skip instead of failing."""
    return os.environ.get(ALLOW_NO_ROM_ENV) == '1'


def require_rom():
    """Fails the calling test when the ROM isn't available (skips it with ``TRIFORCE_ALLOW_NO_ROM=1``)."""
    if not rom_path().exists():
        message = f"{ROM_MISSING} ({rom_path()}); {ALLOW_NO_ROM_ENV}=1 skips the ROM tests"
        if rom_missing_allowed():
            pytest.skip(message)
        pytest.fail(message, pytrace=False)


def require_state(name: str) -> Path:
    """Path of a shipped route start state (``triforce/states/<name>.state``); fails the test
    with the command that builds it when it doesn't exist yet."""
    require_rom()
    path = REPO_STATES_DIR / (name if name.endswith('.state') else f'{name}.state')
    if not path.exists():
        pytest.fail(f"missing start state {name}: build it with `python -m triforce.entry_states build`",
                    pytrace=False)
    return path


def seeded_network(seed: int, config=None):
    """A ``ZeldaPolicy`` whose initial weights depend only on ``seed`` (built under a forked torch RNG, so the
    global RNG is neither read nor advanced).  Without ``config.obs_spec`` it takes the native observation space."""
    # pylint: disable=import-outside-toplevel
    import torch
    from triforce.model import ModelConfig, ZeldaPolicy
    from triforce.observation import observation_space
    config = config if config is not None else ModelConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ZeldaPolicy(None if config.obs_spec else observation_space(), config)


def pytest_configure(config):  # pylint: disable=unused-argument
    """Points ``TRIFORCE_STATES_DIR`` at a fresh session directory before any test module imports ``triforce``, and
    gives torch one thread (``OMP_NUM_THREADS``, inherited by child processes; an explicit value is kept).  The
    tests' networks are tiny: more threads only wait for each other, and on a machine a training round keeps busy
    they made saving and loading the test team 20-40x slower (backlog, Nov 11).  ``TRIFORCE_QUEUE_DIR``: a session
    directory, never the machine's job queue (the training GUI would list the tests' jobs); ``TRIFORCE_JOB_ID=0``:
    the tools the tests call (in-process or as children) run as the child of a job, as under ``triforce job``, so
    they don't refuse to start (``jobs.ENFORCE``); job 0 has no row, so their reports change nothing.  Tests of the
    queue and of the refusal set their own."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    # The job this suite itself runs in (``triforce job test``), noted before the session's own queue replaces it:
    # the suite reports its tests done of collected there (``_report_suite``).
    state = os.environ.get('XDG_STATE_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'state')
    _session['job'] = (os.environ.get('TRIFORCE_JOB_ID'),
                       os.environ.get('TRIFORCE_QUEUE_DIR') or os.path.join(state, 'triforce'))
    _session['queue_dir'] = tempfile.mkdtemp(prefix='triforce-queue-')
    os.environ['TRIFORCE_QUEUE_DIR'] = _session['queue_dir']
    os.environ['TRIFORCE_JOB_ID'] = '0'
    for name in ('TRIFORCE_JOB_LABEL', 'TRIFORCE_JOB_ESTIMATE'):
        os.environ.pop(name, None)
    _session['old_env'] = os.environ.get(STATES_DIR_ENV)
    _session['states_dir'] = Path(tempfile.mkdtemp(prefix='triforce-states-'))
    os.environ[STATES_DIR_ENV] = str(_session['states_dir'])


def _report_suite(force: bool = False):
    """Writes the suite's progress (tests done of collected, tests a second) into the row of the job that runs it;
    at most every two seconds; nothing outside a job, and a queue that can't be written is ignored."""
    job, directory = _session.get('job', (None, None))
    counts = _session.get('tests')
    if not job or job == '0' or counts is None:
        return
    now = time.monotonic()
    if not force and now - _session.get('reported', 0.0) < 2.0:
        return
    _session['reported'] = now
    done, total, began = counts
    # Named for the suite: a test job's bar is the Python suite's alone (the Rust suite beside it reports none).
    report = {'done': done, 'total': total, 'unit': 'Python tests',
              'rate': done / (now - began) if now > began else None, 'ts': time.time()}
    try:
        from triforce import jobqueue  # pylint: disable=import-outside-toplevel
        con = jobqueue.connect(Path(directory))
        try:
            con.execute("UPDATE jobs SET progress = json_set(coalesce(progress, '{}'), '$.tests', json(?)) "
                        "WHERE id = ?", (json.dumps(report), int(job)))
        finally:
            con.close()
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def pytest_collection_finish(session):
    _session['tests'] = [0, len(session.items), time.monotonic()]
    _report_suite(force=True)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):  # pylint: disable=unused-argument
    """Run in parallel (pytest-xdist), the controller collects nothing itself: the workers' collections (the same
    tests each) give the total."""
    if _session.get('tests') is None:
        _session['tests'] = [0, len(ids), time.monotonic()]
        _report_suite(force=True)


def pytest_runtest_logfinish(nodeid, location):  # pylint: disable=unused-argument
    """Every test that ends; in parallel the workers' arrive at the controller, which reports."""
    counts = _session.get('tests')
    if counts is not None:
        counts[0] += 1
        _report_suite(force=counts[0] >= counts[1])


def _controller(config) -> bool:
    """This process hands the tests to xdist's workers and runs none itself."""
    return not hasattr(config, 'workerinput') and (getattr(config.option, 'numprocesses', None) or 0) > 0


def pytest_sessionstart(session):
    """Writes ``start.state`` into the session states dir (without a ROM it stays empty; ``states_dir`` fails),
    and the training tests' task beside it (``start_task``); each xdist worker writes its own, the controller none."""
    if not rom_path().exists() or _controller(session.config):
        return
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist  # pylint: disable=import-outside-toplevel
    game = Game()
    state = game.play_movie(str(BASE_MOVIE))
    assert (state.mode, state.level, state.location) == (5, 0, 0x77)
    game.save_state_file(str(_session['states_dir'] / 'start.state'))
    task = load_specialist('ow-start').replace(start_states=['start'])
    (_session['states_dir'] / START_TASK).write_text(task.to_json(), encoding='utf-8')


START_TASK = 'ow-start-from-start.yaml'


def start_task() -> str:
    """The training tests' task: ``ow-start``'s with ``start`` (the session states dir's one state) as its only seed
    state, a yaml beside it.  Round 40: ``ow-start``'s own globs of shipped states match nothing in the session dir,
    and a training refuses a glob that matches nothing; before, they were dropped and these tests trained on
    ``start`` alone, as they still do."""
    return str(_session['states_dir'] / START_TASK)


def pytest_unconfigure(config):  # pylint: disable=unused-argument
    """Restores ``TRIFORCE_STATES_DIR`` and removes the session states dir and queue dir."""
    if 'states_dir' not in _session:
        return
    if _session['old_env'] is None:
        os.environ.pop(STATES_DIR_ENV, None)
    else:
        os.environ[STATES_DIR_ENV] = _session['old_env']
    shutil.rmtree(_session.pop('states_dir'), ignore_errors=True)
    shutil.rmtree(_session.pop('queue_dir'), ignore_errors=True)


@pytest.fixture(autouse=True)
def logs_dir(tmp_path, monkeypatch):
    """Training event logs go under the test's tmp dir, never the repo's logs/."""
    from triforce.events import LOGS_DIR_ENV  # pylint: disable=import-outside-toplevel
    path = tmp_path / 'logs'
    monkeypatch.setenv(LOGS_DIR_ENV, str(path))
    return path


TEST_TIMEOUT_ENV = 'TRIFORCE_TEST_TIMEOUT'
DEFAULT_TEST_TIMEOUT = 900      # seconds per test; the slowest take about a minute on a loaded machine


@pytest.fixture(autouse=True)
def test_timeout():
    """Fails a test that runs longer than ``$TRIFORCE_TEST_TIMEOUT`` seconds (default ``DEFAULT_TEST_TIMEOUT``; 0
    turns it off), so a hang (a job waiting its turn forever, a child that never answers) ends as a failure with its
    traceback instead of a suite that never ends.  SIGALRM interrupts Python code; a hang inside native code that
    never returns to Python is caught 60 s later by faulthandler, which prints every thread's stack and exits."""
    limit = float(os.environ.get(TEST_TIMEOUT_ENV) or DEFAULT_TEST_TIMEOUT)
    if limit <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def expired(_signum, _frame):
        raise TimeoutError(f"the test ran longer than {limit:.0f} s (${TEST_TIMEOUT_ENV})")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, limit)
    faulthandler.dump_traceback_later(limit + 60, exit=True)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        faulthandler.cancel_dump_traceback_later()
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture(scope='session')
def states_dir():
    """The session states dir holding ``start.state``: every default states dir in this process and its children."""
    require_rom()
    directory = _session['states_dir']
    assert (directory / 'start.state').is_file()
    return directory


@pytest.fixture
def game(states_dir):
    """A fresh native Game per test, with bare state names resolving in ``states_dir``."""
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    return Game(seed=0, states_dir=str(states_dir))


def promote_untrained(root: Path) -> Path:
    """Saves and promotes a small untrained network (``seeded_network(1234 + index)``) for every specialist."""
    # pylint: disable=import-outside-toplevel
    from triforce.model import ModelConfig
    from triforce.registry import promote, save_checkpoint
    from triforce.specialist import list_specialists
    config = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)
    for index, name in enumerate(list_specialists()):
        save_checkpoint(name, 'untrained', seeded_network(1234 + index, config), root=root)
        promote(name, 'untrained', root=root)
    return root


@pytest.fixture(scope='module')
def models_dir(tmp_path_factory):
    """Every specialist promoted to a small untrained network, the same weights whatever ran before."""
    return promote_untrained(tmp_path_factory.mktemp('models'))


# A chain segment a scripted network wins whatever the seed: from the shipped state 0_3cc_0 (0x3c, where Link
# comes out of Dungeon 2, route step 24) straight south is 0x4c (route step 25), ow-octorok's room all the way.
SEGMENT_STATE = '0_3cc_0'
SEGMENT_START_STEP = 24
SEGMENT_STOP_STEP = 25


@pytest.fixture(scope='session')
def scripted_models(tmp_path_factory):
    """``models_dir``'s untrained team with ``ow-octorok`` promoted to a network that walks south
    (``networks.scripted('MOVE S 16')``): it wins the ``SEGMENT_STATE`` segment."""
    # pylint: disable=import-outside-toplevel
    from triforce.registry import promote, save_checkpoint
    from .networks import scripted
    root = promote_untrained(tmp_path_factory.mktemp('scripted-models'))
    save_checkpoint('ow-octorok', 'south', scripted('MOVE S 16'), root=root)
    promote('ow-octorok', 'south', root=root)
    return root


@pytest.fixture(scope='session')
def recordings(scripted_models, states_dir, tmp_path_factory):  # pylint: disable=redefined-outer-name
    """A folder of three replays (read only): two wins of the scripted segment (seeds 0 and 1) and a run from
    power on that ends on its step budget (seed 2)."""
    # pylint: disable=import-outside-toplevel
    from triforce import replay
    from triforce.runner import Route, Runner
    folder = tmp_path_factory.mktemp('recordings')
    runner = Runner(models_dir=scripted_models, max_steps=20, parallel=2, states_dir=states_dir)
    segment = Route.load('d2first', start_from=str(require_state(SEGMENT_STATE)), stop_step=SEGMENT_STOP_STEP)
    record = replay.record_options(runner.record_meta(segment))
    runner.run_many(segment, [0, 1], record=record, replays=replay.ReplayDir(folder))
    boot = Route.load('d2first', power_on={'delay': 40})
    runner.run_many(boot, [2], record=replay.record_options(runner.record_meta(boot)),
                    replays=replay.ReplayDir(folder))
    assert sorted(p.name for p in folder.iterdir()) == [
        '00000000-win.tfreplay', '00000001-win.tfreplay', '00000002-stuck.tfreplay']
    return folder
