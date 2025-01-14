import traceback
from multiprocessing import Process, Queue

class PPOSubprocess:
    """Controls a worker process that runs PPO in a separate process."""

    def __init__(self, idx, create_env, ppo_class, ppo_kwargs, result_queue):
        """
        idx: worker index
        create_env: callable that returns an environment
        ppo_class: the PPO class (or a factory function) for local instantiation
        ppo_kwargs: dictionary of arguments to init local PPO
        """
        self.idx = idx

        # The queues for communication
        self.command_queue = Queue()

        # Store references we need in the child process.
        # (We won't call these directly in the main process.)
        self._create_env = create_env
        self._ppo_class = ppo_class
        self._ppo_kwargs = ppo_kwargs

        # Create and start the worker process.
        args = idx, create_env, ppo_class, ppo_kwargs,self.command_queue, result_queue
        self.process = Process(target=self._run, args=args)
        self.process.start()

    def _run(self, idx, create_env, ppo_class, ppo_kwargs, command_queue, result_queue):
        ppo = ppo_class(**ppo_kwargs)
        env = create_env()

        # pylint: disable=broad-except
        iteration = 0
        try:
            while True:
                command = command_queue.get()

                match command['command']:
                    case 'exit':
                        message = { 'idx' : idx, 'command' : 'exit' }
                        result_queue.put(message)
                        break

                    case 'build_batch':
                        ppo.network.load_state_dict(command['weights'])
                        infos, next_value = ppo.build_one_batch(0, env, command.get('progress', None), iteration)
                        iteration += 1

                        message = {
                            'idx' : idx,
                            'command' : 'build_batch',
                            'infos' : infos,
                            'next_value' : next_value,
                            'observation' : ppo.observation,
                            'dones' : ppo.dones,
                            'act_logp_ent_val' : ppo.act_logp_ent_val,
                            'masks' : ppo.masks,
                            'rewards' : ppo.rewards
                            }

                        result_queue.put(message)

        except Exception as e:
            message = {
                'idx' : idx,
                'command' : 'error',
                'error' : e,
                'traceback' : traceback.format_exc()
                }

            result_queue.put(message)

    def build_batch_async(self, weights, progress):
        """Ask the worker to build a batch; returns immediately."""
        message = {
            'command': 'build_batch',
            'progress': progress,
            'weights': weights
            }

        self.command_queue.put(message)

    def close_async(self):
        """Ask the worker to exit; returns immediately."""
        self.command_queue.put({'command' : 'exit'})
        self.process.join()

    def join(self):
        """Wait for the worker to exit."""
        self.process.join()
