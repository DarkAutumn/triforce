import traceback
from multiprocessing import Process, Queue

from .ml_ppo_rollout_buffer import PPORolloutBuffer
from .models import Network, create_network

class SubprocessWorker:
    """Controls a worker process that runs PPO in a separate process."""
    def __init__(self, idx, create_env, network, result_queue, ppo_kwargs):
        self.idx = idx

        # The queues for communication
        self.command_queue = Queue()

        # Store references we need in the child process.
        # (We won't call these directly in the main process.)
        self._create_env = create_env
        self._ppo_kwargs = ppo_kwargs

        # Create and start the worker process.
        self.process = Process(target=self.run, args=(idx, create_env, network, ppo_kwargs,
                                                      self.command_queue, result_queue))
        self.process.start()

    def run(self, idx, create_env, network : Network, ppo_kwargs, command_queue, result_queue):
        """The main loop of the worker process."""
        env = create_env()
        network = create_network(network, env.observation_space, env.action_space)

        # pylint: disable=broad-except
        buffer = None
        try:
            while True:
                command = command_queue.get()
                match command['command']:
                    case 'exit':
                        message = { 'idx' : idx, 'command' : 'exit' }
                        result_queue.put(message)
                        break

                    case 'build_batch':
                        if buffer is None:
                            buffer = PPORolloutBuffer(ppo_kwargs['steps'], 1, env.observation_space, env.action_space,
                                                        ppo_kwargs['gamma'], ppo_kwargs['lambda'])

                        network.load_state_dict(command['weights'])
                        infos = buffer.ppo_main_loop(0, network, env, None)
                        result_queue.put({
                            'idx' : idx,
                            'command' : 'build_batch',
                            'infos' : infos,
                            'result' : buffer,
                            })

        except Exception as e:
            result_queue.put({
                'idx' : idx,
                'command' : 'error',
                'error' : e,
                'traceback' : traceback.format_exc()
                })

    def run_main_loop_async(self, weights):
        """Ask the worker to build a batch; returns immediately."""
        self.command_queue.put({
            'command': 'build_batch',
            'weights': weights
            })

    def close_async(self):
        """Ask the worker to exit; returns immediately."""
        msg = { 'command' : 'exit' }
        self.command_queue.put(msg)

    def join(self):
        """Wait for the worker to exit."""
        self.close_async() # multiple calls are ok
        self.process.join()
