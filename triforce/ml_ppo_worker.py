"""Subprocess worker for parallel PPO rollout collection."""

import multiprocessing as mp

from .ml_ppo_rollout_buffer import PPORolloutBuffer

# Commands sent from main process to worker
CMD_COLLECT = 'collect'
CMD_UPDATE_WEIGHTS = 'update_weights'
CMD_CLOSE = 'close'


class EnvFactory:
    """Picklable factory for creating Zelda environments in worker subprocesses."""
    def __init__(self, scenario_def, action_space, **kwargs):
        self.scenario_def = scenario_def
        self.action_space = action_space
        self.kwargs = kwargs

    def __call__(self):
        from .zelda_env import make_zelda_env  # pylint: disable=import-outside-toplevel
        return make_zelda_env(self.scenario_def, self.action_space, **self.kwargs)


class SimpleEnvFactory:
    """Picklable factory that creates an env from a class and args."""
    def __init__(self, env_class, *args, **kwargs):
        self.env_class = env_class
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.env_class(*self.args, **self.kwargs)


def _worker_loop(conn, create_env_fn, network_class, target_steps, gamma, lam):
    """Main loop for a rollout worker subprocess."""
    # Each worker only does single-sample inference — restrict PyTorch to 1 thread
    # to avoid massive thread contention across N worker processes.
    import torch  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(1)

    env = create_env_fn()
    try:
        buffer = PPORolloutBuffer(target_steps, 1, env.observation_space, env.action_space, gamma, lam)
        network = network_class(env.observation_space, env.action_space)

        while True:
            cmd, data = conn.recv()

            if cmd == CMD_UPDATE_WEIGHTS:
                network.load_state_dict(data)
                network.eval()
                conn.send(('ready', None))

            elif cmd == CMD_COLLECT:
                buffer.ppo_main_loop(0, network, env, None)
                steps = buffer.memory_length

                # Send buffer data back as serializable tensors
                buf_data = _extract_buffer_data(buffer)
                conn.send(('rollout', (buf_data, steps)))

            elif cmd == CMD_CLOSE:
                conn.send(('closed', None))
                break
    finally:
        env.close()
        conn.close()


def _extract_buffer_data(buffer):
    """Extracts tensor data from a single-env buffer for transfer to main process."""
    data = {
        'dones': buffer.dones.clone(),
        'act_logp_ent_val': buffer.act_logp_ent_val.clone(),
        'rewards': buffer.rewards.clone(),
        'masks': buffer.masks.clone(),
        'returns': buffer.returns.clone(),
        'advantages': buffer.advantages.clone(),
        'has_data': buffer.has_data,
    }
    if isinstance(buffer.observation, dict):
        data['observation'] = {k: v.clone() for k, v in buffer.observation.items()}
    else:
        data['observation'] = buffer.observation.clone()
    return data


def _apply_buffer_data(target_buffer, env_index, buf_data):
    """Applies received buffer data into the target multi-env buffer at env_index."""
    target_buffer.dones[env_index] = buf_data['dones'][0]
    target_buffer.act_logp_ent_val[env_index] = buf_data['act_logp_ent_val'][0]
    target_buffer.rewards[env_index] = buf_data['rewards'][0]
    target_buffer.masks[env_index] = buf_data['masks'][0]
    target_buffer.returns[env_index] = buf_data['returns'][0]
    target_buffer.advantages[env_index] = buf_data['advantages'][0]

    if isinstance(target_buffer.observation, dict):
        for key in target_buffer.observation:
            target_buffer.observation[key][env_index] = buf_data['observation'][key][0]
    else:
        target_buffer.observation[env_index] = buf_data['observation'][0]

    target_buffer.has_data = True


class RolloutWorkerPool:
    """Manages a pool of subprocess workers for parallel rollout collection."""
    def __init__(self, n_workers, create_env_fn, network_class, target_steps, gamma, lam):
        self.n_workers = n_workers
        self.workers = []
        self.conns = []

        ctx = mp.get_context('spawn')
        for _ in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker_loop,
                args=(child_conn, create_env_fn, network_class, target_steps, gamma, lam),
                daemon=True
            )
            process.start()
            child_conn.close()
            self.conns.append(parent_conn)
            self.workers.append(process)

    def update_weights(self, network):
        """Sends updated network weights to all workers."""
        state_dict = {k: v.cpu() for k, v in network.state_dict().items()}

        for conn in self.conns:
            conn.send((CMD_UPDATE_WEIGHTS, state_dict))

        for conn in self.conns:
            msg, _ = conn.recv()
            assert msg == 'ready'

    def collect_rollouts(self, target_buffer, progress=None):
        """Collects rollouts from all workers into the target multi-env buffer."""
        # Dispatch collection to all workers
        for conn in self.conns:
            conn.send((CMD_COLLECT, None))

        # Collect results
        total_steps = 0
        for i, conn in enumerate(self.conns):
            msg, data = conn.recv()
            assert msg == 'rollout'
            buf_data, steps = data
            _apply_buffer_data(target_buffer, i, buf_data)
            total_steps += steps

        if progress:
            progress.update(total_steps)

        return total_steps

    def close(self):
        """Shuts down all workers."""
        for conn in self.conns:
            try:
                conn.send((CMD_CLOSE, None))
            except BrokenPipeError:
                pass

        for conn in self.conns:
            try:
                conn.recv()
            except (BrokenPipeError, EOFError):
                pass

        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        self.conns.clear()
        self.workers.clear()
