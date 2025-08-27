import time
import torch
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from RL4CRN.environments.environment import Environment

class AbstractMultiEnvironments:
    """
    Abstract class for multiple environments in CRN reinforcement learning.
    This class provides a framework for managing multiple CRN environments in parallel.
    """
    def __init__(self, envs, hall_of_fame_size=10, logger=None):
        """
        Initialize the parallel environments with a list of environments and an optional logger.
        Args:
            envs (list): A list of environment instances.
            logger (Logger, optional): An optional logger for logging metrics.
        """
        self.envs = envs
        self.logger = logger
        self.rendering_iteration = 0
        self.hall_of_fame_size = hall_of_fame_size
        self.hall_of_fame = [Environment(envs[0].state.clone(), envs[0].max_added_reactions, logger=logger, logger_schedule=1) for _ in range(hall_of_fame_size)]
        self.hall_of_fame_empty = True

    def reset(self):
        """
        Reset all environments to their initial state.
        Returns:
            list: A list of initial states for each environment.
        """
        return [env.reset() for env in self.envs]
    
    def gather(self):
        """
        Gather the current state from all environments.
        Returns:
            list: A list of current states (IOCRNs) for each environment.
        """
        return [env.state for env in self.envs]
    
    def step(self, actions, mode='reaction index'):
        """
        Step through all environments with the provided actions.
        Args:
            actions (list): A list of actions (reactions) to be taken in each environment.
            mode (str): The mode of action interpretation, e.g., 'complex index', 'reaction index', or 'species index'.
        Returns:
            list: A list of tuples containing the new state and done flag for each environment.
        """
        tic_step = time.time()
        output = [env.step(action, mode) for env,action in zip(self.envs, actions)]
        toc_step = time.time()
        if self.logger is not None:
            self.logger.log_metric('Step Time', toc_step - tic_step)
        return output
    
    def observe(self):
        """
        Observe the current state of all environments.
        Returns:
            tuple: A tuple containing:
                - reactions_indices_batch: A numpy array representing the reactions indices in the batch of IOCRNs. Shape: (N, m).
                - rate_constants_batch: A numpy array representing the reaction rate constants in the batch of IOCRNs. Shape: (N, m).
                - reactions_indices_influenced_by_inputs_batch: A list of p numpy arrays, each containing the influenced reactions for a specific input. 
                Each numpy array is associated with a specific input and has shape (N, #), where # is the maximum number of reactions in any CRN in the batch influenced by this input.
        """
        reactions_indices_batch = np.array([env.state.reactions_indices for env in self.envs])
        rate_constants_batch = np.array([env.state.c for env in self.envs])
        reactions_indices_influenced_by_inputs_batch = []
        for i in range(self.envs[0].state.p):
            rows = [np.array(env.state.list_influenced_reactions[i]) for env in self.envs]
            max_len = max((len(r) for r in rows), default=0)
            padded = [np.pad(r, (0, max_len - len(r)), constant_values=0) for r in rows]
            reactions_indices_influenced_by_inputs_batch.append(np.array(padded).astype(np.int64))
        return reactions_indices_batch, rate_constants_batch, reactions_indices_influenced_by_inputs_batch
    
    def render(self, rewards, n_best=1, disregarded_percentage=0.9, mode={'style': 'logger', 'task': 'transients', 'format': 'figure'}):
        """
        Render the current state of the environments based on the provided mode.
        Args:
            rewards (list): A list of rewards for each environment.
            n_best (int): The number of best environments to render every step.
            disregarded_percentage (float): The percentage of environments to disregard when selecting the best ones.
            mode (dict): A dictionary specifying the rendering style and task.
                - 'style': 'human' for interactive display, 'logger' for logging to a file.
                - 'task': 'transients' for transient response, 'rank' for ranking.
                - 'format': 'figure' or 'image' for the output format.
        """
        if mode['style'] == 'logger':
            if self.logger is not None:
                # Collect the top_k environments (indices) based on the rewards
                top_k = torch.topk(torch.tensor(rewards), int(len(rewards) * (1.-disregarded_percentage)), largest=False).indices
                self.rendering_iteration += 1

                # Render the n_best environments
                for i in range(n_best):
                    self.envs[top_k[i]].render(mode=mode, ID=f'{self.rendering_iteration}_{i}')

                # Render the hall of fame
                for i in range(self.hall_of_fame_size):
                    self.hall_of_fame[i].render(mode=mode, ID = f'hof_{i}')
                
                # Render the top_k environments based on the specified mode
                match mode:
                    case {'style': 'logger', 'task': 'transients'}:
                        fig, axes = plt.subplots(self.envs[0].state.q, 1, figsize=(10, 5 * self.envs[0].state.q))
                        if not isinstance(axes, (list, np.ndarray)):
                            axes = [axes]
                        for i in top_k:
                            fig, axes = self.envs[i].state.plot_transient_response(fig, axes)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN Distribution {self.rendering_iteration}')
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration} (Top {(1.-disregarded_percentage)*100}%)', figure=fig)
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