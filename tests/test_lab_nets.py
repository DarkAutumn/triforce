"""The lab's networks (``triforce.nets``): the head-logits interface, serving torch networks to the pool, other
architectures side by side with ``ZeldaPolicy`` (docs/lab.md)."""

import numpy as np
import pytest
import torch

from triforce import _native, nets
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
        def answer(slots, logits, values):
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
        def answer(slots, logits, values):
            answered.append((logits.copy(), values.copy()))

    server.queue = Queue()
    n = observations['action_mask'].shape[0]
    server.answer({'obs': observations, 'slot': np.arange(n), 'model': np.zeros(n, dtype=np.int64),
                   'job': np.arange(n), 'step': np.zeros(n, dtype=np.int64), 'bootstrap': np.zeros(n, dtype=bool)})
    (logits, values), = answered
    heads, _ = network.cpu().heads({k: torch.as_tensor(v) for k, v in observations.items()})
    tolerance = 5e-2 if server.half else 1e-5
    assert np.allclose(logits, heads.flat().detach().numpy(), atol=tolerance)
    assert np.allclose(values, heads.value.detach().numpy(), atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need the GPU")
def test_cuda_graphs_answer_as_the_eager_forward_and_fall_back_when_they_disagree(observations, monkeypatch, capsys):
    """A graphed answer is the eager forward's for every batch size (rows padded to the bucket), a replaced network
    drops its graphs, and a graph that disagrees with the eager forward turns the network eager for good."""
    torch.manual_seed(5)
    networks = [nets.create('wide', native_spec(), hidden_dim=64, depth=2) for _ in range(2)]
    answered = []

    class Queue:
        logits_size = 11 * 13

        @staticmethod
        def answer(slots, logits, values):
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
    finally:
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
