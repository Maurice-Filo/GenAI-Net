import time
import torch
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from RL4CRN.environments.environment import Environment
from RL4CRN.utils.visualizations import topology_graph

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
        """ Reset all environments to their initial state.
        Returns:
            list: A list of initial states for each environment. """
        
        return [env.reset() for env in self.envs]
    
    def gather(self):
        """
        Gather the current state from all environments.
        Returns:
            list: A list of current states (IOCRNs) for each environment.
        """
        return [env.state for env in self.envs]
    
    def step(self, actions, stepper):
        """
        Step through all environments with the provided actions.
        Args:
            actions (list): A list of actions (reactions) to be taken in each environment.
            stepper: 
        Returns:
            list: A list of tuples containing the new state and done flag for each environment.
        """
        tic_step = time.time()
        output = [env.step(action, stepper) for env,action in zip(self.envs, actions)]
        toc_step = time.time()
        if self.logger is not None:
            self.logger.log_metric('Step Time', toc_step - tic_step)
        return output
    
    def observe(self, observer, tensorizer):
        output = [observer.observe(env.state) for env in self.envs]
        tensorized_output = torch.stack([tensorizer.tensorize(o) for o in output])
        return tensorized_output.float()
    
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
        tic_step = time.time()
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

                # Render the IOCRN diversity graph
                if mode.get('topology', True):
                    iocrn_list = []
                    for idx in top_k:
                        iocrn_list.append(self.envs[idx].state)
                    fig_graph = topology_graph(iocrn_list, t=10, figsize = (10,10))
                    fig_graph1 = topology_graph(iocrn_list, t=3, figsize = (10,10))
                    buf = BytesIO()
                    fig_graph.savefig(buf, format='png')
                    buf.seek(0)
                    self.logger.log_image(buf, name=f'CRN Diversity Graph {self.rendering_iteration}')
                    buf.close()
                    buf = BytesIO()
                    fig_graph1.savefig(buf, format='png')
                    buf.seek(0)
                    self.logger.log_image(buf, name=f'CRN Diversity Graph (Clusters) {self.rendering_iteration}')
                    buf.close()
                    plt.close(fig_graph)
                    plt.close(fig_graph1)
                
                # Render the top_k environments based on the specified mode
                match mode:
                    case {'style': 'logger', 'task': 'transients'}:
                        fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes, (list, np.ndarray)):
                            axes = [axes]
                        for i in top_k:
                            fig, axes = self.envs[i].state.plot_transient_response(fig, axes)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN Distribution {self.rendering_iteration}')
                        bounds = mode.get('bounds')
                        if bounds is not None:  
                            for a, b in zip(axes, bounds):
                                a.set_ylim([0, b])
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

                    case {'style': 'logger', 'task': 'transients + dose-response'}:
                        fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes, (list, np.ndarray)):
                            axes = [axes]
                        for i in top_k:
                            fig, axes = self.envs[i].state.plot_transient_response(fig, axes)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN Distribution {self.rendering_iteration}')
                        bounds = mode.get('bounds')
                        if bounds is not None:
                            for a, b in zip(axes, bounds):
                                a.set_ylim([0, b])
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

                        fig1, axes1 = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes1, (list, np.ndarray)):
                            axes1 = [axes1]
                        for i in top_k:
                            fig1, axes1 = self.envs[i].state.plot_dose_response(fig1, axes1)
                        fig1.tight_layout(rect=[0, 0, 1, 0.95])
                        fig1.suptitle(f'CRN Distribution {self.rendering_iteration} Dose Response')
                        if bounds is not None:
                            for a, b in zip(axes1, bounds):
                                a.set_ylim([0, b])
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration} Dose Reponse (Top {(1.-disregarded_percentage)*100}%)', figure=fig1)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig1.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN Distribution {self.rendering_iteration} Dose Response')
                            buf.close()
                        else:
                            raise ValueError(f"Unknown mode: {mode[1]}. Use 'figure' or 'image'.")
                        plt.close(fig1)

                    case {'style': 'logger', 'task': 'phase_plot'}:
                        fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
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

                        if self.envs[0].state.num_species == 2:
                            fig1, axis1 = plt.subplots(figsize=(10, 10))
                        elif self.envs[0].state.num_species == 3:
                            fig1 = plt.figure(figsize=(10, 10))
                            axis1 = fig1.add_subplot(111, projection='3d')
                        for i in top_k:
                            fig1, axis1 = self.envs[i].state.plot_phase_portrait(fig1, axis1)
                        fig1.tight_layout(rect=[0, 0, 1, 0.95])
                        fig1.suptitle(f'CRN Distribution Phase Portrait {self.rendering_iteration}')
                        bounds = mode.get('bounds')
                        if bounds is not None:
                            axis1.set_xlim([0, bounds[0]])
                            axis1.set_ylim([0, bounds[1]])
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN Distribution Phase Portrait {self.rendering_iteration} (Top {(1.-disregarded_percentage)*100}%)', figure=fig1)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig1.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN Distribution Phase Portrait {self.rendering_iteration}')
                            buf.close()
                        else:
                            raise ValueError(f"Unknown mode: {mode[1]}. Use 'figure' or 'image'.")
                        plt.close(fig1)

                    case {'style': 'logger', 'task': 'rank'}:
                        ranks = np.array([env.state.last_task_info['rank'] for env in self.envs])
                        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
                        axes.hist(ranks, bins=range(1, ranks.max().item() + 2), align='left', rwidth=0.8)
                        axes.set_xlabel('Rank')
                        axes.set_ylabel('Frequency')
                        axes.set_title('Histogram of Stoichiometry Matrix Rank')
                        self.logger.log_figure(figure_name=f'Stoichiometry Matrix Rank Distribution {self.rendering_iteration}', figure=fig)
                        plt.close(fig)
                    
                    case {'style': 'logger', 'task': 'transients + frequency content'}:
                        fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes, (list, np.ndarray)):
                            axes = [axes]
                        for i in top_k:
                            fig, axes = self.envs[i].state.plot_transient_response(fig, axes)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN Distribution {self.rendering_iteration}')
                        bounds = mode.get('bounds')
                        if bounds is not None:
                            for a, b in zip(axes, bounds):
                                a.set_ylim([0, b])
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

                        fig1, axes1 = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes1, (list, np.ndarray)):
                            axes1 = [axes1]
                        for i in top_k:
                            fig1, axes1 = self.envs[i].state.plot_frequency_content(fig1, axes1, t0=mode.get('t0', 0.0))
                        fig1.tight_layout(rect=[0, 0, 1, 0.95])
                        fig1.suptitle(f'CRN Distribution {self.rendering_iteration} Frequency Content')
                        bounds_freq = mode.get('bounds_freq')
                        if bounds_freq is not None:
                            for a, b in zip(axes1, bounds_freq):
                                if b[0] is not None:
                                    a.set_xlim([0, b[0]])
                                if b[1] is not None:
                                    a.set_ylim([0, b[1]])
                                if mode.get('scale', 'linear') == 'log':
                                    a.set_yscale('log')
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration} Frequency Content (Top {(1.-disregarded_percentage)*100}%)', figure=fig1)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig1.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN Distribution {self.rendering_iteration} Frequency Content')
                            buf.close()
                        else:
                            raise ValueError(f"Unknown mode: {mode[1]}. Use 'figure' or 'image'.")
                        plt.close(fig1)

                    case _:
                        raise ValueError(f"Unknown mode: {mode}. Check the spelling!")
                    
        toc_step = time.time()
        if self.logger is not None:
            self.logger.log_metric('Render Time', toc_step - tic_step)