"""The lab's networks (``triforce.nets``): the head-logits interface, serving torch networks to the pool, other
architectures side by side with ``ZeldaPolicy`` (docs/lab.md)."""

import numpy as np
import pytest
import torch

from triforce import _native, nets, replay
from triforce.evaluate import evaluate_specialist
from triforce.nets.base import Heads, LabNet, factored_evaluate, factored_joint
from triforce.nets import serve as serve_module
from triforce.nets.serve import Server
from triforce.registry import load_model, save_checkpoint

from .conftest import REPO_STATES_DIR, require_state
from .networks import native_spec, random_network

ACT = 'temperature:0.5+relax:16'


@pytest.fixture(scope='module')
def observations():
    """Real observations of a few rooms, played by a random Rust policy."""
    network = random_network(3)
    native = network.to_native()
    rows = []
    for state, family in (('0_0as_0', 'ow-lynel'), ('3_5bs_0', 'darknut'), ('2_0es_0', 'boss-dodongo')):
        require_state(state)
        config = _native.SpecialistConfig.load(family).replace(state_routes=['d2first'])
        env = _native.Env(config, states_dir=str(REPO_STATES_DIR), seed=3)
        obs, _ = env.reset(seed=3, options={'start_state': state})
        for step in range(20):
            rows.append(obs)
            batch = {k: np.asarray(v)[None] for k, v in obs.items()}
            action = native.evaluate(batch, noise=_native.keyed_noise([7], [step], native.noise_size))['actions'][0]
            obs, _, done, truncated, _ = env.step([int(a) for a in action])
            if done or truncated:
                break
    return {k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}


def test_heads_reproduce_zelda_policy_evaluate_and_joint(observations):
    network = random_network(1)
    heads, state = network.heads(observations)
    assert state is None
    mask = torch.as_tensor(observations['action_mask'])
    with torch.no_grad():
        for deterministic in (True, False):
            noise = torch.as_tensor(_native.keyed_noise(list(range(mask.shape[0])), [0] * mask.shape[0], 11 + 8 + 4))
            want = network.evaluate(observations, deterministic=deterministic, noise=noise)
            got = factored_evaluate(heads, mask, deterministic=deterministic, noise=noise)
            assert torch.equal(got.actions, want.actions)
            assert torch.allclose(got.log_prob, want.log_prob, atol=1e-6)
            assert torch.allclose(got.entropy, want.entropy, atol=1e-6)
        assert torch.allclose(factored_joint(heads, mask), network.joint_log_probs(observations), atol=1e-6)


def test_rust_logits_match_torch_heads(observations):
    network = random_network(2)
    logits, values = network.to_native().logits(observations)
    heads, _ = network.heads(observations)
    assert np.allclose(logits, heads.flat().detach().numpy(), atol=2e-5)
    assert np.allclose(values, heads.value.detach().numpy(), atol=2e-5)


@pytest.mark.parametrize('settings', [{'hidden_dim': 512, 'entity_layers': 2},
                                      {'hidden_dim': 1024, 'tower_depth': 6, 'entity_layers': 2},
                                      {'tower_depth': 4, 'entity_d_model': 128, 'entity_ff_dim': 256,
                                       'grid_channels': [32, 64, 64, 64], 'grid_out_dim': 256}])
def test_the_rust_actor_plays_the_capacity_shapes(observations, settings):
    """Wider and deeper ``ZeldaPolicy``s (a capacity test's sizes) run on the Rust path: same logits and values as
    torch. A depth of 3 isn't written into ``model_config`` (older code reads the checkpoint as before)."""
    torch.manual_seed(5)
    network = nets.create('zelda', native_spec(), **settings)
    assert ('tower_depth' in network.config.to_dict()) == ('tower_depth' in settings)
    logits, values = network.to_native().logits(observations)
    heads, _ = network.heads(observations)
    assert np.allclose(logits, heads.flat().detach().numpy(), atol=5e-5)
    assert np.allclose(values, heads.value.detach().numpy(), atol=5e-5)


class RustLogits(torch.nn.Module):
    """Serves a Rust ``Policy``'s own logits: acting from them is the Rust actor bit for bit
    (``policy::logits``), so a served play must match the Rust play run for run."""

    def __init__(self, native):
        super().__init__()
        self.native = native
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = nets.base.NetConfig({'arch': 'rust-logits', 'obs_spec': {}})

    def heads(self, obs, state=None):
        logits, values = self.native.logits({k: v.cpu().numpy() for k, v in obs.items()})
        logits = torch.from_numpy(logits)
        k, d, u = 11, 8, 4
        return Heads(logits[:, :k], logits[:, k:k + k * d].view(-1, k, d), logits[:, k + k * d:].view(-1, k, u),
                     torch.from_numpy(values)), state


def _play(pool_task, policy, act, parallel=3):
    pool = _native.EnvPool(pool_task, num_envs=parallel, threads=parallel, seed=1)
    try:
        pool.play([policy], act=act)
        records = []
        while True:
            idle = pool.wait_idle(timeout=0.1)
            records += [e['record'] for e in pool.poll_events() if e['type'] == 'eval']
            if idle:
                return sorted(records, key=lambda r: r['seed'])
    finally:
        pool.close()


@pytest.mark.parametrize('act', ['sample', ACT, 'argmax'])
def test_a_served_network_plays_run_for_run_like_the_rust_actor(states_dir, act):
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=40)
    task = _native.EvalTask(config, [('start', seed) for seed in range(6)], states_dir=str(states_dir))
    native = random_network(4).to_native()
    server = Server(device='cpu')
    try:
        served = _play(task, server.policy(RustLogits(native)), act)
    finally:
        server.close()
    rust = _play(task, native, act)
    assert len(rust) == 6
    for a, b in zip(rust, served):
        a, b = dict(a), dict(b)
        assert a == b


class HitLogits(RustLogits):
    """``RustLogits`` with a hit head: joint action ``j``'s hit logit is ``j / 88 - 2``."""

    def heads(self, obs, state=None):
        heads, state = super().heads(obs, state)
        heads.aux = {'hit': (torch.arange(352, dtype=torch.float32) / 88 - 2).expand(heads.kind.shape[0], -1)}
        return heads, state


@pytest.mark.parametrize('head', [RustLogits, HitLogits])
def test_a_served_hit_head_is_recorded_as_each_decisions_danger(states_dir, head):
    """The server answers the hit head's logits beside the others; the replay's DECISIONS keep P(hit) per legal joint
    action (a byte: within 1/255) for a network with one, and no danger for one without."""
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=12)
    task = _native.EvalTask(config, [('start', 0)], states_dir=str(states_dir))
    server = Server(device='cpu')
    pool = _native.EnvPool(task, num_envs=1, threads=1, seed=1, record=replay.record_options(replay.meta('eval', [])))
    try:
        pool.play([server.policy(head(random_network(4).to_native()))], act='sample')
        data = []
        while True:
            idle = pool.wait_idle(timeout=0.1)
            data += [e['data'] for e in pool.poll_events() if e['type'] == 'replay']
            if idle:
                break
    finally:
        pool.close()
        server.close()
    decisions = _native.Replay(data[0]).decisions()
    assert len(decisions['frame']) > 0
    if head is RustLogits:
        assert all(row is None for row in decisions['danger'])
        return
    for row in decisions['danger']:
        assert row
        for joint, p in row.items():
            assert abs(p - 1 / (1 + np.exp(2 - joint / 88))) <= 1 / 255 + 1e-6


class Counter(LabNet):
    """A recurrent toy: its state counts the decisions of the episode; the kind logits prefer kind ``count % 2``."""
    arch = 'test-counter'
    defaults = {}

    def __init__(self, config):
        super().__init__(config)
        self.scale = torch.nn.Parameter(torch.ones(1))

    def initial_state(self, batch):
        return torch.zeros(batch, 1)

    def heads(self, obs, state=None):
        batch = obs['action_mask'].shape[0]
        count = state[:, 0]
        kind = torch.zeros(batch, 11)
        kind[torch.arange(batch), (count % 2).long()] = 10.0
        return (Heads(kind * self.scale, torch.zeros(batch, 11, 8), torch.zeros(batch, 11, 4), count.clone()),
                state + 1)


def test_the_server_keeps_a_recurrent_state_per_env_and_restarts_it_per_episode():
    server = Server(device='cpu')
    server.models[0] = Counter({'obs_spec': native_spec()})
    answered = []

    class Queue:        # stands in for the native queue: keeps the answers
        logits_size = 11 * 13

        @staticmethod
        def answer(slots, logits, values, hit):
            answered.append(values.copy())

    server.queue = Queue()
    obs = {'action_mask': np.ones((2, 11 * 13), dtype=bool)}

    def ask(steps, jobs, bootstrap=(False, False)):
        server.answer({'obs': obs, 'slot': np.array([0, 1]), 'model': np.array([0, 0]), 'job': np.array(jobs),
                       'step': np.array(steps), 'bootstrap': np.array(bootstrap)})
        return list(answered[-1])

    assert ask([0, 0], [1, 2]) == [0, 0]
    assert ask([1, 1], [1, 2]) == [1, 1]
    assert ask([-1, 2], [-1, 2], bootstrap=(True, False)) == [2, 2]     # a bootstrap reads, doesn't advance
    assert ask([2, 0], [1, 3]) == [2, 0]                                # slot 1: a new episode starts over
    assert ask([3, 1], [1, 3]) == [3, 1]


@pytest.mark.parametrize('half', [False, True])
def test_a_served_batch_comes_back_in_one_copy_and_half_precision_only_on_the_gpu(observations, half):
    """The server's answer (logits and value, one copy back) is the network's; ``half`` runs it in float16 on the GPU
    (within the half's precision) and is ignored on the CPU."""
    device = 'cuda' if half and torch.cuda.is_available() else 'cpu'
    server = Server(device=device, half=half)
    assert server.half == (device == 'cuda')
    torch.manual_seed(3)
    network = nets.create('wide', native_spec(), hidden_dim=64, depth=2)
    server.models[0] = network.to(server.device)
    answered = []

    class Queue:
        logits_size = 11 * 13

        @staticmethod
        def answer(slots, logits, values, hit):
            answered.append((logits.copy(), values.copy(), hit.copy()))

    server.queue = Queue()
    n = observations['action_mask'].shape[0]
    server.answer({'obs': observations, 'slot': np.arange(n), 'model': np.zeros(n, dtype=np.int64),
                   'job': np.arange(n), 'step': np.zeros(n, dtype=np.int64), 'bootstrap': np.zeros(n, dtype=bool)})
    (logits, values, hit), = answered
    heads, _ = network.cpu().heads({k: torch.as_tensor(v) for k, v in observations.items()})
    tolerance = 5e-2 if server.half else 1e-5
    assert np.allclose(logits, heads.flat().detach().numpy(), atol=tolerance)
    assert np.allclose(values, heads.value.detach().numpy(), atol=tolerance)
    assert hit.shape == (n, 352) and np.isnan(hit).all(), "no hit head: a row of NaN"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need the GPU")
def test_cuda_graphs_answer_as_the_eager_forward_and_fall_back_when_they_disagree(observations, monkeypatch, capsys):
    """A graphed answer is the eager forward's for every batch size (rows padded to the bucket), a replaced network
    drops its graphs, and a graph that disagrees with the eager forward turns the network eager for good."""
    # A chain's server: no learner in this process (a lab training test before this one in the worker marks it one).
    monkeypatch.setattr(serve_module, '_LEARNER', {'on': False, 'graphs': False})
    torch.manual_seed(5)
    networks = [nets.create('wide', native_spec(), hidden_dim=64, depth=2) for _ in range(2)]
    answered = []

    class Queue:
        logits_size = 11 * 13

        @staticmethod
        def answer(slots, logits, values, hit):
            answered.append((logits.copy(), values.copy()))

    def ask(server, n):
        obs = {k: v[:n] for k, v in observations.items()}
        server.answer({'obs': obs, 'slot': np.arange(n), 'model': np.zeros(n, dtype=np.int64), 'job': np.arange(n),
                       'step': np.zeros(n, dtype=np.int64), 'bootstrap': np.zeros(n, dtype=bool)})
        return answered[-1]

    graphed, eager = Server(device='cuda', graphs=True), Server(device='cuda', graphs=False)
    for server in (graphed, eager):
        server.queue = Queue()
        server.models[0] = networks[0].to('cuda').eval()
    total = observations['action_mask'].shape[0]
    for n in sorted({1, 3, min(5, total), total}):
        for _ in range(5):          # past the checked replays
            (a, va), (b, vb) = ask(graphed, n), ask(eager, n)
            assert np.allclose(a, b, atol=1e-5) and np.allclose(va, vb, atol=1e-5)
    assert graphed._graphs and not graphed._eager
    # A replaced network: ``policy`` drops the key's graphs; this one's replays disagree with its eager forward.
    graphed.models[0] = eager.models[0] = networks[1].to('cuda').eval()
    graphed._graphs.clear()
    monkeypatch.setattr(serve_module._Graph, 'run', lambda self, obs: self.output + 1.0)
    (a, _), (b, _) = ask(graphed, 3), ask(eager, 3)
    assert np.allclose(a, b, atol=1e-5) and 0 in graphed._eager       # answered eagerly, and eager from now on
    assert 'runs without CUDA graphs' in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need the GPU")
def test_a_learner_process_serves_eagerly_beside_its_own_gpu_work(observations, monkeypatch):
    """A training's main thread recomputes values on the GPU while the server answers the rollouts (Nov 23: a graph
    capture on the server thread and the learner's forwards failed each other, in every capture mode): a learner
    process (``serve.learner_process``) serves eagerly, and both threads finish."""
    import threading
    monkeypatch.setattr(serve_module, '_LEARNER', {'on': False, 'graphs': False})
    assert Server(device='cuda').graphs
    serve_module.learner_process()
    torch.manual_seed(7)
    network = nets.create('wide', native_spec(), hidden_dim=64, depth=2).to('cuda').eval()
    other = nets.create('wide', native_spec(), hidden_dim=64, depth=2).to('cuda').eval()
    batch = {k: torch.as_tensor(v).to('cuda') for k, v in observations.items()}
    stop, errors = threading.Event(), []

    def learner():
        try:
            with torch.no_grad():
                while not stop.is_set():
                    other.heads(batch)
                    torch.cuda.synchronize()
        except Exception as error:     # pylint: disable=broad-exception-caught
            errors.append(error)
    thread = threading.Thread(target=learner)
    thread.start()
    try:
        server = Server(device='cuda')
        server.models[0] = network

        class Queue:
            logits_size = 11 * 13

            @staticmethod
            def answer(slots, logits, values, hit):
                pass
        server.queue = Queue()
        for n in (1, 2, 3, 5, 8):
            obs = {k: v[:n] for k, v in observations.items()}
            server.answer({'obs': obs, 'slot': np.arange(n), 'model': np.zeros(n, dtype=np.int64),
                           'job': np.arange(n), 'step': np.zeros(n, dtype=np.int64),
                           'bootstrap': np.zeros(n, dtype=bool)})
    finally:
        stop.set()
        thread.join()
    assert not server.graphs and not errors and not server._graphs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need the GPU")
def test_a_lab_trainer_captures_every_bucket_at_its_first_batch_then_replays_beside_its_learner(observations,
                                                                                                monkeypatch):
    """The lab trainer (``learner_process(graphs=True)``, round40.md "Where a teacher's time goes"): the network's
    every batch bucket is captured at its first batch, while nothing else uses the GPU; later batches replay beside
    the learner's own GPU work on another thread and answer as the eager forward does; a later rollout's
    ``serve_copy`` copies the new weights into the same served copy (its graphs kept and reading them), and never
    captures again.  ``TRIFORCE_SERVE_GRAPHS=0`` still serves eagerly."""
    import threading
    monkeypatch.setattr(serve_module, '_LEARNER', {'on': False, 'graphs': False})
    serve_module.learner_process(graphs=True)
    torch.manual_seed(11)
    learner_net = nets.create('wide', native_spec(), hidden_dim=64, depth=2).to('cuda')
    answered = []

    class Queue:
        logits_size = 11 * 13

        @staticmethod
        def answer(slots, logits, values, hit):
            answered.append(logits.copy())

        @staticmethod
        def policy(index, weights):
            return index

    def ask(server, n):
        obs = {k: v[:n] for k, v in observations.items()}
        server.answer({'obs': obs, 'slot': np.arange(n), 'model': np.zeros(n, dtype=np.int64), 'job': np.arange(n),
                       'step': np.zeros(n, dtype=np.int64), 'bootstrap': np.zeros(n, dtype=bool)})
        return answered[-1]

    server = Server(device='cuda')
    assert server.graphs and server.capture_all and server.gather == serve_module.GRAPH_GATHER
    server.queue = Queue()
    server._start = lambda: None
    server.serve_copy(learner_net, key='student')
    served = server.models[0]
    first = ask(server, 3)                       # the first rollout: every bucket captured now
    assert {size for _, size in server._graphs} == set(serve_module.GRAPH_BUCKETS)
    eager = Server(device='cuda', graphs=False)
    eager.queue = Queue()
    eager.models[0] = served
    assert np.allclose(first, ask(eager, 3), atol=1e-5)
    batch = {k: torch.as_tensor(v).to('cuda') for k, v in observations.items()}
    stop, errors = threading.Event(), []

    def learner():
        try:
            optimizer = torch.optim.Adam(learner_net.parameters(), lr=1e-2)
            while not stop.is_set():
                heads = learner_net.heads(batch)[0]
                loss = heads.flat().pow(2).mean() + heads.value.pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
        except Exception as error:     # pylint: disable=broad-exception-caught
            errors.append(error)
    thread = threading.Thread(target=learner)
    thread.start()
    try:
        graphs = dict(server._graphs)
        for n in (1, 2, 5, 8, 3):                # replays beside the learner: no capture, no failure
            assert np.allclose(ask(server, n), ask(eager, n), atol=1e-5)
    finally:
        stop.set()
        thread.join()
    assert not errors and not server._eager and server._graphs == graphs
    # The next rollout: the learner's new weights copied into the same served copy, its graphs kept and current.
    with torch.no_grad():
        for parameter in learner_net.parameters():
            parameter.add_(0.05)
    server.serve_copy(learner_net, key='student')
    assert server.models[0] is served and server._graphs == graphs
    assert all(torch.equal(a, b) for a, b in zip(served.state_dict().values(), learner_net.state_dict().values()))
    assert not np.allclose(ask(server, 3), first, atol=1e-5) and np.allclose(ask(server, 3), ask(eager, 3), atol=1e-5)
    monkeypatch.setenv('TRIFORCE_SERVE_GRAPHS', '0')
    assert not Server(device='cuda').graphs and serve_module.serve_mode()['forward'] == 'eager'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need the GPU")
def test_a_lab_trainers_server_stages_one_copy_a_batch_and_answers_no_hit_logits(observations, monkeypatch):
    """The lab trainer's server (round40.md "Where a teacher's time goes", the second step): a batch's observation
    arrays go to the GPU as one copy from a pinned buffer (``_Graph`` ``staged``), and it answers no hit logits (a
    training never reads them); the logits and values are the eager forward's, staged or not, for every batch size.
    A chain's server (no learner) answers the hit logits; ``TRIFORCE_SERVE_HITS=1`` keeps them in a training."""
    monkeypatch.setattr(serve_module, '_LEARNER', {'on': False, 'graphs': False})
    eager = Server(device='cuda', graphs=False)
    serve_module.learner_process(graphs=True)
    torch.manual_seed(13)
    network = nets.create('wide', native_spec(), hidden_dim=64, depth=2).to('cuda').eval()

    def server(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        made = Server(device='cuda')
        for key in env:
            monkeypatch.delenv(key)
        return made
    servers = {'staged': server(), 'unstaged': server(TRIFORCE_SERVE_STAGING='0'),
               'hits': server(TRIFORCE_SERVE_HITS='1'), 'eager': eager}
    assert servers['staged'].staged and not servers['staged'].hits
    assert not servers['unstaged'].staged and servers['hits'].hits and eager.hits
    answered = {}
    for name, each in servers.items():
        class Queue:            # pylint: disable=too-few-public-methods
            logits_size = 11 * 13

            @staticmethod
            def answer(slots, logits, values, hit, name=name):
                answered[name] = (logits.copy(), values.copy(), None if hit is None else hit.copy())
        each.queue = Queue()
        each.models[0] = network
    total = observations['action_mask'].shape[0]
    for n in sorted({1, 3, min(5, total), total}):
        for _ in range(4):          # past the checked replays
            obs = {k: v[:n] for k, v in observations.items()}
            for each in servers.values():
                each.answer({'obs': obs, 'slot': np.arange(n), 'model': np.zeros(n, dtype=np.int64),
                             'job': np.arange(n), 'step': np.zeros(n, dtype=np.int64),
                             'bootstrap': np.zeros(n, dtype=bool)})
            for name in ('staged', 'unstaged', 'hits'):
                assert np.allclose(answered[name][0], answered['eager'][0], atol=1e-5)
                assert np.allclose(answered[name][1], answered['eager'][1], atol=1e-5)
            assert answered['staged'][2] is None and answered['unstaged'][2] is None
            assert answered['hits'][2].shape == answered['eager'][2].shape == (n, serve_module.JOINT)
    assert all(each._graphs and not each._eager for name, each in servers.items() if name != 'eager')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need the GPU")
def test_a_capture_that_runs_out_of_gpu_memory_serves_eagerly_and_the_run_goes_on(observations, monkeypatch,
                                                                                   capsys):
    """A training whose first capture cannot get its GPU memory (the restart's three trainings beside a suite, the
    lead's question of Nov 25): the network is served eagerly from then on with one warning line, the answers are the
    eager forward's, and the GPU stays usable (the capture ended)."""
    monkeypatch.setattr(serve_module, '_LEARNER', {'on': False, 'graphs': False})
    eager = Server(device='cuda', graphs=False)
    serve_module.learner_process(graphs=True)
    torch.manual_seed(17)
    network = nets.create('wide', native_spec(), hidden_dim=64, depth=2).to('cuda').eval()
    heads, calls = network.heads, []

    def short_of_memory(obs, state=None):
        calls.append(torch.cuda.is_current_stream_capturing())
        if calls[-1]:
            raise torch.OutOfMemoryError("CUDA out of memory (the test's)")
        return heads(obs, state)
    monkeypatch.setattr(network, 'heads', short_of_memory)
    server = Server(device='cuda')
    answered = {}
    for name, each in (('server', server), ('eager', eager)):
        class Queue:            # pylint: disable=too-few-public-methods
            logits_size = 11 * 13

            @staticmethod
            def answer(slots, logits, values, hit, name=name):
                answered[name] = logits.copy()
        each.queue = Queue()
        each.models[0] = network
    for n in (3, 5, 1):
        obs = {k: v[:n] for k, v in observations.items()}
        for each in (server, eager):
            each.answer({'obs': obs, 'slot': np.arange(n), 'model': np.zeros(n, dtype=np.int64), 'job': np.arange(n),
                         'step': np.zeros(n, dtype=np.int64), 'bootstrap': np.zeros(n, dtype=bool)})
        assert np.allclose(answered['server'], answered['eager'], atol=1e-5)
    assert any(calls) and 0 in server._eager and not server._graphs
    assert not torch.cuda.is_current_stream_capturing()
    assert (torch.ones(4, device='cuda') * 2).sum().item() == 8.0
    warnings = [line for line in capsys.readouterr().out.splitlines() if 'runs without CUDA graphs' in line]
    assert len(warnings) == 1 and 'OutOfMemoryError' in warnings[0] and 'the run goes on' in warnings[0]


def test_networks_served_from_temporaries_never_share_an_index():
    """``load_model(path)[0].to_native()`` in a loop (``runner.Policies``): each temporary dies at once, and a later
    network may reuse its id; each must keep its own index and weights (Nov 22: chains with several served
    networks played one family's rooms with another's network in about two processes of three)."""
    import gc
    server = Server(device='cpu')
    policies, sums = [], []
    for seed in range(6):
        torch.manual_seed(seed)
        network = nets.create('wide', native_spec(), hidden_dim=32, depth=1)
        sums.append(sum(float(p.detach().abs().sum()) for p in network.parameters()))
        policies.append(server.serve_copy(network))
        del network
        gc.collect()
    try:
        assert len(server.models) == 6 and not server._keys      # every temporary gone, its key forgotten
        served = [sum(float(p.detach().abs().sum()) for p in server.models[i].parameters()) for i in range(6)]
        assert served == sums
        learner = nets.create('wide', native_spec(), hidden_dim=32, depth=1)
        server.serve_copy(learner)
        server.serve_copy(learner)                                  # a live network's rollouts reuse one index
        assert len(server.models) == 7 and server._keys == {id(learner): 6}
        policies.clear()                                            # the chain ends: its networks are freed
        gc.collect()
        server._collect()
        assert list(server.models) == [6] and sorted(server._free) == list(range(6))
    finally:
        server.close()


def test_a_bench_loop_of_served_networks_stays_flat(states_dir, observations):
    """``evaluate_specialist`` loads its checkpoint per bench entry and serves a copy (``to_native``): every copy
    and its CUDA graphs were kept, and Stage52's job #2579 (a 768-wide hitnet) grew to 12 GB and died of CUDA OOM
    after 5320 episodes. Once a network and its policies are gone its index is freed and reused: the served set
    and the GPU memory stay flat, every load answers a recorded batch bit for bit as the first and plays the same
    actions (a play's float sums move in the last digit with how the envs' rows fall into batches)."""
    import gc
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=30)
    task = _native.EvalTask(config, [('start', seed) for seed in range(2)], states_dir=str(states_dir))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    server = Server(device=device, graphs=device == 'cuda')
    torch.manual_seed(7)
    saved = nets.create('wide', native_spec(), hidden_dim=64, depth=2).state_dict()
    n = observations['action_mask'].shape[0]
    batch = {'obs': observations, 'slot': np.arange(n), 'model': np.zeros(n, dtype=np.int64), 'job': np.arange(n),
             'step': np.zeros(n, dtype=np.int64), 'bootstrap': np.zeros(n, dtype=bool)}
    answers, plays, memory = [], [], []

    @torch.no_grad()
    def answer():
        """The served network's answer to the recorded batch, as the server computes it (its graph when it has one;
        no request is in flight: the server thread waits for one)."""
        network = server.models[0]
        out = server._graphed(0, network, batch, np.arange(n)) if server.graphs else None
        if out is None:
            heads, _ = network.heads({k: torch.from_numpy(v).to(server.device) for k, v in observations.items()})
            out = serve_module._flat(heads, server.hits).cpu().numpy()
        return out

    try:
        for _ in range(8):
            network = nets.create('wide', native_spec(), hidden_dim=64, depth=2)
            network.load_state_dict(saved)
            policy = server.serve_copy(network)
            assert policy.model == 0 and server.queue.leases(0) == 1
            answers.append(answer())
            records = _play(task, policy, 'argmax', parallel=2)
            plays.append([(r['outcome'], [v['decisions']['actions'] for v in r['visits']]) for r in records])
            del network, policy
            gc.collect()
            server._collect()
            assert not server.models and not server._graphs and not server._states and not server._keys
            assert server.queue.leases(0) == 0 and server._free == [0]
            if device == 'cuda':
                memory.append(torch.cuda.memory_allocated())
        assert all(np.array_equal(a, answers[0], equal_nan=True) for a in answers)    # no hit head: NaN hit rows
        assert len(plays[0]) == 2 and all(play == plays[0] for play in plays)
        assert len(set(memory)) <= 1, memory
    finally:
        server.close()


def test_a_collection_inside_a_collection_frees_each_index_once():
    """Dropping a freed network may run a finalizer that collects again on the same thread (a dead network's
    ``_forget``): each index is freed once, never twice into ``_free``, and two live networks never share one."""
    import weakref
    server = Server(device='cpu')
    try:
        for index in range(3):
            dropped = torch.nn.Linear(1, 1)
            weakref.finalize(dropped, server._collect)          # runs inside the collection that drops it
            server.models[index] = dropped
            del dropped
        server._orphans, server._next = {0, 1, 2}, 3
        server._collect()
        assert sorted(server._free) == [0, 1, 2] and not server._orphans and not server.models
        live = [nets.create('wide', native_spec(), hidden_dim=32, depth=1) for _ in range(4)]
        indices = [server.serve_copy(network).model for network in live]
        assert sorted(indices) == [0, 1, 2, 3] and not server._free
    finally:
        server.close()


def test_a_freed_index_waits_for_every_policy_and_pool_that_holds_it(states_dir):
    """A network whose key died stays served while an ``ExternalPolicy`` of it lives, and while a pool plays with
    it after that policy is gone (the actor holds its own lease); a new network meanwhile gets another index."""
    import gc
    import threading
    server = Server(device='cpu')
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=20)
    task = _native.EvalTask(config, [('start', seed) for seed in range(2)], states_dir=str(states_dir))
    pool = _native.EnvPool(task, num_envs=2, threads=2, seed=1)
    gate = threading.Event()
    answer = server.answer
    server.answer = lambda batch: (gate.wait(), answer(batch))
    try:
        network = nets.create('wide', native_spec(), hidden_dim=32, depth=1)
        kept = server.serve_copy(network)
        del network
        gc.collect()
        server._collect()
        assert server._orphans == {0} and 0 in server.models          # the policy still holds it
        assert server.serve_copy(other := nets.create('wide', native_spec(), hidden_dim=32, depth=1)).model == 1
        pool.play([kept], act='argmax')                                # its first request waits at the gate
        del kept
        gc.collect()
        server._collect()
        assert server.queue.leases(0) == 1 and 0 in server.models     # the pool's actor holds it
        gate.set()
        records = []
        while not pool.wait_idle(timeout=0.1):
            records += [e['record'] for e in pool.poll_events() if e['type'] == 'eval']
        records += [e['record'] for e in pool.poll_events() if e['type'] == 'eval']
        assert len(records) == 2 and server.error is None
        pool.close()
        gc.collect()
        server._collect()
        assert server._free == [0] and 0 not in server.models and server.queue.leases(0) == 0
        assert server.serve_copy(nets.create('wide', native_spec(), hidden_dim=32, depth=1)).model == 0
        assert server._keys[id(other)] == 1
    finally:
        gate.set()
        pool.close()
        server.close()


def test_architectures_side_by_side_save_load_and_play(states_dir, tmp_path):
    assert nets.names()[0] == 'zelda' and {'wide', 'gru', 'hithead'} <= set(nets.names())
    wide = nets.create('wide', native_spec(), hidden_dim=64, depth=2)
    zelda = nets.create('zelda', native_spec())
    assert nets.arch_of(wide.config.to_dict()) == 'wide' and 'arch' not in zelda.config.to_dict()
    for name, network in (('wide', wide), ('zelda', zelda)):
        path = save_checkpoint('ow-start', name, network, root=tmp_path)
        loaded, checkpoint = load_model(path)
        assert type(loaded) is type(network)
        assert checkpoint['model_config'] == network.config.to_dict()
        for key, value in network.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)
    with pytest.raises(ValueError):
        nets.create('wide', native_spec(), width=3)
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=20)
    for name in ('wide', 'zelda'):
        result = evaluate_specialist(config, tmp_path / 'ow-start' / name, episodes_per_state=3, parallel=3, seed=2,
                                     states_dir=states_dir, act=ACT, clips='on')
        assert result['episodes'] == 3 and all(0 < r['steps'] <= 20 for r in result['records'])
