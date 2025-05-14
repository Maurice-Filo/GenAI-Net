import time
import torch
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

class AbstractVecCRNEnvironment:
    """
    Abstract class for vectorized CRN environments.
    This class is designed to be inherited by specific implementations of vectorized CRN environments.
    It provides a common interface for interacting with multiple CRN environments in parallel.
    """
    def __init__(self, envs, logger=None):
        # Initialize the environment with a list of CRN environments and an optional logger.
        self.envs = envs
        self.logger = logger
        self.rendering_iteration = 0

    def reset(self):
        return [env.reset() for env in self.envs]
    
    def gather(self):
        return [env.state for env in self.envs]
    
    def step(self, actions, mode='reaction index'):
        tic_step = time.time()
        output = [env.step(action, mode) for env,action in zip(self.envs, actions)]
        toc_step = time.time()
        if self.logger is not None:
            self.logger.log_metric('Step Time', toc_step - tic_step)
        return output
    
    def render(self, rewards, n_best=1, disregarded_percentage=0.9, mode={'style': 'logger', 'task': 'transients', 'format': 'figure'}):
        if mode['style'] == 'logger':
            if self.logger is not None:
                top_k = torch.topk(torch.tensor(rewards), int(len(rewards) * (1.-disregarded_percentage)), largest=False).indices
                self.rendering_iteration += 1
                for i in range(n_best):
                    self.envs[top_k[i]].render(mode=mode, ID = f'{self.rendering_iteration}_{i}')

                match mode:
                    case {'style': 'logger', 'task': 'transients'}:
                        fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes, (list, np.ndarray)):
                            axes = [axes]
                        for i in top_k:
                            fig, axes = self.envs[i].state.plot_transient_response(fig, axes)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN Distribution {self.rendering_iteration}')
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration}', figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN Distribution {self.rendering_iteration}')
                            buf.close()
                        else:
                            raise ValueError(f"Unknown mode: {mode[1]}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    case {'style': 'logger', 'task': 'rank'}:
                        ranks = np.array([env.state.last_task_info['rank'] for env in self.envs])
                        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
                        axes.hist(ranks, bins=range(1, ranks.max().item() + 2), align='left', rwidth=0.8)
                        axes.set_xlabel('Rank')
                        axes.set_ylabel('Frequency')
                        axes.set_title('Histogram of Stoichiometry Matrix Rank')
                        self.logger.log_figure(figure_name=f'Stoichiometry Matrix Rank Distribution {self.rendering_iteration}', figure=fig)
                        plt.close(fig)
                    case _:
                        raise ValueError(f"Unknown mode: {mode}. Check the spelling!")
            
    def observe(self):
        reactions_indices_batch = np.array([env.state.reactions_indices for env in self.envs])
        parameters_batch = np.array([env.state.parameters for env in self.envs])
        reactions_indices_influenced_by_inputs_batch = []
        for i in range(self.envs[0].state.num_inputs):
            rows = [np.array(env.state.list_influenced_reactions[i]) for env in self.envs]
            max_len = max((len(r) for r in rows), default=0)
            padded = [np.pad(r, (0, max_len - len(r)), constant_values=0) for r in rows]
            reactions_indices_influenced_by_inputs_batch.append(np.array(padded).astype(np.uint64))
        return reactions_indices_batch, parameters_batch, reactions_indices_influenced_by_inputs_batch