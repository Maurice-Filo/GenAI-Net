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
    
    def step(self, actions):
        tic_step = time.time()
        output = [env.step(action) for env,action in zip(self.envs, actions)]
        toc_step = time.time()
        if self.logger is not None:
            self.logger.log_metric('Step Time', toc_step - tic_step)
        return output
    
    def render(self, rewards, n_best=1, disregarded_percentage=0.9, mode='logger_figure'):
        mode = mode.split('_')
        if mode[0] == 'logger':
            if self.logger is not None:
                top_k = torch.topk(torch.tensor(rewards), int(len(rewards) * (1.-disregarded_percentage)), largest=False).indices
                self.rendering_iteration += 1
                for i in range(n_best):
                    self.envs[top_k[i]].render(mode=mode, ID = f'{self.rendering_iteration}_{i}')
                fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                if not isinstance(axes, (list, np.ndarray)):
                    axes = [axes]
                for i in top_k:
                    fig, axes = self.envs[i].state.plot_transient_response(fig, axes)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.suptitle(f'CRN Distribution {self.rendering_iteration}')
                if mode[1] == 'figure':
                    self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration}', figure=fig)
                elif mode[1] == 'image':
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    self.logger.log_image(buf, name=f'CRN Distribution {self.rendering_iteration}')
                    buf.close()
                else:
                    raise ValueError(f"Unknown mode: {mode[1]}. Use 'figure' or 'image'.")
                plt.close(fig)