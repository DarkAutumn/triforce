from multiprocessing import Queue, Process

class PPOSubprocess:
    """
    A persistent worker process that owns:
      - A single environment (from create_env())
      - A local PPO(n_envs=1)
      - Its own carry-over (obs, done, action_mask) state
    It listens on a queue for commands like:
      - 'build_batch': run build_one_batch, compute_returns, etc.
      - 'update_weights': update local PPO state_dict
      - 'close': terminate
    """

    def __init__(self, idx, create_env, ppo_class, ppo_kwargs):
        """
        idx: worker index
        create_env: callable that returns an environment
        ppo_class: the PPO class (or a factory function) for local instantiation
        ppo_kwargs: dictionary of arguments to init local PPO
        """
        self.idx = idx
        self.command_queue = Queue()
        self.result_queue = Queue()

        # We'll create a separate Process that runs self._run()
        self.process = Process(target=self._run, args=(create_env, ppo_class, ppo_kwargs))
        self.process.start()

    def _run(self, create_env, ppo_class, ppo_kwargs):
        """
        The target method running inside the worker process.
        """
        # 1) Create environment and local PPO(n_envs=1)
        env = create_env()
        local_ppo = ppo_class(**ppo_kwargs)  # e.g. PPO(network=..., device=..., n_envs=1, ...)
        local_ppo.n_envs = 1  # ensure single env in the worker

        # 2) Maintain carry-over state for the environment
        #    None means "we haven't started yet" => we reset in build_one_batch
        worker_state = None

        # 3) Loop, waiting for commands
        while True:
            cmd, data = self.command_queue.get()
            if cmd == 'build_batch':
                # data might be (iterations, progress, state_dict) or similar
                # if you want to sync weights:
                if 'weights' in data and data['weights'] is not None:
                    local_ppo.load_state_dict(data['weights'])

                progress = data.get('progress', None)
                iterations = data.get('iterations', 0)

                # Run build_one_batch + compute_returns
                # TODO: push returns/advs to the main process
                # TODO: implement loading/saving weights
                i = self.idx  # worker index
                infos, next_value, worker_state = local_ppo.build_one_batch(i, env, progress, worker_state)


                # Send results back
                self.result_queue.put((infos, next_value, local_ppo.obs, local_ppo.dones,
                                       local_ppo.act_logp_ent_val, local_ppo.masks, local_ppo.rewards))

            elif cmd == 'update_weights':
                # data == new_state_dict
                local_ppo.load_state_dict(data)
                self.result_queue.put("weights_updated")

            elif cmd == 'close':
                # Clean up
                env.close()
                break  # exit the loop => process ends

            else:
                print(f"[Worker {self.idx}] Unknown command: {cmd}")
                self.result_queue.put(None)

    def build_batch_async(self, iterations=None, progress=None, weights=None):
        """
        Asynchronously request that the worker build a batch of data.
        weights: optionally pass in main PPO's state_dict if you want to sync.
        """
        self.command_queue.put((
            'build_batch',
            {
                'iterations': iterations,
                'progress': progress,
                'weights': weights,
            }
        ))

    def get_result(self):
        """Blocking call to retrieve the last result from the worker."""
        return self.result_queue.get()

    def update_weights_async(self, new_weights):
        """Send a message to update the local PPO's weights."""
        self.command_queue.put(('update_weights', new_weights))

    def close(self):
        """Close the worker process."""
        self.command_queue.put(('close', None))
        self.process.join()
