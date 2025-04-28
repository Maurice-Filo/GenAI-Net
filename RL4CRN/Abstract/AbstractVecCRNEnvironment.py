import time
import torch
import matplotlib.pyplot as plt

class AbstractCRNEnvironment:
    def __init__(self, envs, logger=None):
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
    
    def render(self, rewards, n_best=1, risk=0.9, mode='logger'):
        if mode == 'logger':
            if self.logger is not None:
                top_k = torch.topk(torch.tensor(rewards), int(len(rewards) * (1.-risk)), largest=False).indices
                self.rendering_iteration += 1
                for i in range(n_best):
                    self.envs[top_k[i]].render(mode=mode, ID = f'{self.rendering_iteration}_{i}')
                fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                for i in top_k:
                    fig, axes = self.envs[i].state.plot_reactions(fig, axes)
                self.logger.log_figure(fig, f'CRN Distribution {self.rendering_iteration}')