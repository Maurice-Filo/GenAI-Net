"""
Multi-environment utilities for CRN reinforcement learning.

This module defines `AbstractMultiEnvironments`, a lightweight manager for
running multiple CRN environments in parallel (synchronously) and providing
common operations such as reset, stepping, observation/tensorization, and rich
rendering/logging of the best-performing environments.

Key features:
    - **Batch stepping**: applies one action per environment.
    - **Observation pipeline**: uses an observer and tensorizer to produce a batch
      tensor suitable for an agent.
    - **Logging-oriented rendering**: selects top-performing environments and
      logs plots/images to a provided logger.
    - **Hall of fame (optional)**: keeps a buffer of top environments and renders
      them alongside the current batch.

Notes:
    This class is intentionally "abstract" in the sense that it does not enforce a
    specific environment implementation beyond the methods accessed in the code
    (`reset`, `step`, `render`, and `state` access). It can be used directly if the
    provided environments follow the expected interface.
"""

import time
import torch
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from RL4CRN.environments.environment import Environment
from RL4CRN.utils.visualizations import topology_graph
from RL4CRN.utils.visualizations import plot_truth_table
from RL4CRN.utils.hall_of_fame import HallOfFame

class AbstractMultiEnvironments:
    """Synchronous manager for multiple CRN environments.

    Args:
        envs: List of environment instances. Each environment is expected to
            expose:
            - `reset()`
            - `step(action, stepper, raw_action=None)` (raw_action optional)
            - `render(mode=..., ID=...)`
            - `state` attribute (IOCRN-like), used by observers and plotting code
        hall_of_fame_size: Maximum number of environments stored in the hall of
            fame. If 0, hall of fame is disabled.
        logger: Optional logger for metrics and figures/images. Expected to
            provide methods such as `log_metric`, `log_figure`, and `log_image`.

    Attributes:
        envs: List of managed environments.
        hall_of_fame: Optional `RL4CRN.utils.hall_of_fame.HallOfFame`
            storing best-performing environments.
        rendering_iteration: Counter used to label logged figures/images.
    """
    def __init__(self, envs, hall_of_fame_size=10, logger=None):
        self.envs = envs
        self.logger = logger
        self.rendering_iteration = 0
        if hall_of_fame_size > 0:
            self.hall_of_fame = HallOfFame(max_size=hall_of_fame_size)
        else:
            self.hall_of_fame = None

    def reset(self):
        """Reset all environments.

        Returns:
            List of initial states for each environment (as returned by each
                environment's `reset()` method).
        """
        return [env.reset() for env in self.envs]
    
    def gather(self):
        """Gather the current state from all environments.

        Returns:
            List of current environment states (typically IOCRN objects), one per
                environment.
        """
        return [env.state for env in self.envs]
    
    def step(self, actions, stepper, raw_actions=None):
        """Step all environments forward by one action.

        Args:
            actions: List of environment actions to apply, one per environment.
            stepper: Stepper object passed to each environment's `step(...)`.
            raw_actions: Optional list of raw policy actions aligned with `actions`.
                If provided, each environment is called as
                `env.step(action, stepper, raw_action=raw_action)`.

        Returns:
            List of per-environment step outputs. The structure depends on the
                underlying environment's `step` method (commonly `(state, done)` or
                similar).

        Side Effects:
            Logs a timing metric `'Timing: Step'` if a logger is available.
        """
        tic_step = time.time()
        if raw_actions is None:
            output = [env.step(action, stepper) for env,action in zip(self.envs, actions)]
        else:
            output = [env.step(action, stepper, raw_action=raw_action) for env,action,raw_action in zip(self.envs, actions, raw_actions)]
        toc_step = time.time()
        if self.logger is not None:
            self.logger.log_metric('Timing: Step', toc_step - tic_step)
        return output
    
    def observe(self, observer, tensorizer):
        """Observe all environments and tensorize the resulting observations.

        Args:
            observer: Observer providing `observe(state)` and returning an
                observation object per environment.
            tensorizer: Tensorizer providing `tensorize(observation)` and returning
                a tensor per environment (or at least stackable tensors).

        Returns:
            Tensor of stacked observations with shape `(N, ...)`, where `N` is the
                number of environments.
        """
        output = [observer.observe(env.state) for env in self.envs]
        tensorized_output = torch.stack([tensorizer.tensorize(o) for o in output])
        return tensorized_output.float()
    
    def render(self, rewards, n_best=1, disregarded_percentage=0.9, mode={'style': 'logger', 'task': 'transients', 'format': 'figure'}):
        r"""Render and/or log diagnostics for the current batch of environments.

        This method is primarily designed for logging workflows. It selects a set
        of top environments according to `rewards` and logs a variety of plots,
        depending on the requested `mode`.

        Selection of top environments:
            Rewards are interpreted as *losses* (smaller is better), and the
            top subset is selected via:

            $$k = \left\lfloor N (1 - p) \right\rfloor.$$

            where $p$ is `disregarded_percentage` and $N$ is the number
            of environments.

        Args:
            rewards: Sequence of per-environment scalar scores. Smaller values are
                treated as better when selecting top environments.
            n_best: Number of best environments to render individually via each
                environment's `render(...)`.
            disregarded_percentage: Fraction of environments to discard when forming
                the "top-k" subset for aggregate plots. For example, with
                `disregarded_percentage=0.9`, only the best 10% are considered.
            mode: Dictionary describing the rendering behavior. Expected keys:
                - `style`: `'logger'` (supported here) or `'human'` (not implemented
                  in this method).
                - `task`: controls what diagnostics to log. Supported values in this
                  method include (depending on code paths):
                  `'transients'`, `'transients + dose-response'`, `'phase_plot'`,
                  `'rank'`, `'transients + frequency content'`, `'transients + logic'`,
                  `'SSA_transients'`.
                - `format`: `'figure'` to log matplotlib figures directly, or `'image'`
                  to log PNG buffers.
                Additional optional keys may be used by specific tasks:
                - `bounds`, `bounds_freq`, `scale`, `t0`, `topology`.

        Returns:
            None.

        Side Effects:
            - Calls `env.render(...)` on selected environments.
            - Logs figures/images/metrics via `self.logger` (if present).
            - May create and close matplotlib figures.
            - Increments `self.rendering_iteration` on each call in logger mode.
        """
        tic_step = time.time()
        if mode['style'] == 'logger':
            if self.logger is not None:
                # Collect the top_k environments (indices) based on the rewards
                top_k = torch.topk(torch.tensor(rewards), int(len(rewards) * (1.-disregarded_percentage)), largest=False).indices
                self.rendering_iteration += 1

                # Render the n_best environments
                for i in range(min(n_best, len(top_k))):
                    self.envs[top_k[i]].render(mode=mode, ID=f'{self.rendering_iteration}_{i}')

                # Render the hall of fame
                if self.hall_of_fame is not None:
                    for i, env in enumerate(self.hall_of_fame):
                        env.render(mode=mode, ID = f'hof_{i}')
                    
                    hof_iocrn_list = [env.state for env in self.hall_of_fame]
                    fig_graph = topology_graph(hof_iocrn_list, t=5, figsize = (10,10))
                    buf = BytesIO()
                    fig_graph.savefig(buf, format='png')
                    buf.seek(0)
                    self.logger.log_image(buf, name=f'HOF Diversity Graph {self.rendering_iteration}')
                    buf.close()

                # Render the IOCRN diversity graph
                if mode.get('topology', True):
                    iocrn_list = []
                    for idx in top_k:
                        iocrn_list.append(self.envs[idx].state)
                    fig_graph = topology_graph(iocrn_list, t=5, figsize = (10,10))
                    # fig_graph1 = topology_graph(iocrn_list, t=3, figsize = (10,10))
                    buf = BytesIO()
                    fig_graph.savefig(buf, format='png')
                    buf.seek(0)
                    self.logger.log_image(buf, name=f'CRN Diversity Graph {self.rendering_iteration}')
                    buf.close()
                    # buf = BytesIO()
                    # fig_graph1.savefig(buf, format='png')
                    # buf.seek(0)
                    # self.logger.log_image(buf, name=f'CRN Diversity Graph (Clusters) {self.rendering_iteration}')
                    # buf.close()
                    # plt.close(fig_graph)
                    # plt.close(fig_graph1)
                
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

                    
                    
                    case {'style': 'logger', 'task': 'transients + logic'}:
                        # --- 1. Transients Ensemble (Overlayed) ---
                        # Initialize figure
                        fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes, (list, np.ndarray)):
                            axes = [axes]
                        
                        # Loop through top_k and overlay plots
                        for i in top_k:
                            # Assuming plot_transient_response can take existing fig/axes arguments 
                            # and plot with alpha (transparency)
                            fig, axes = self.envs[i].state.plot_transient_response(fig=fig, axes=axes, alpha=0.2)
                        
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN Distribution {self.rendering_iteration} (Top {(1.-disregarded_percentage)*100}%)')
                        
                        # Apply bounds if provided
                        bounds = mode.get('bounds')
                        if bounds is not None:
                            for a, b in zip(axes, bounds):
                                if b is not None: a.set_ylim([0, b])

                        # Log Transients
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration} Transients', figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN Distribution {self.rendering_iteration} Transients')
                            buf.close()
                        else:
                            raise ValueError(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)

                        # --- 2. Truth Tables (All Top-K) ---
                        try:

                            # Iterate through ALL top performing CRNs to show their logic
                            for rank, env_idx in enumerate(top_k):
                                current_state = self.envs[env_idx].state
                                reward_val = current_state.last_task_info['reward']
                                
                                logic_inputs = current_state.last_task_info.get('inputs')
                                logic_outputs = current_state.last_task_info.get('outputs') 
                                
                                if logic_inputs is not None and logic_outputs is not None:
                                    # Handle 3D time-series data (extract steady state)
                                    # e.g. shape (1000, 1, 16) or (16, 1000, 1) -> take last time point
                                    raw_out = np.array(logic_outputs)
                                    if raw_out.ndim == 3:
                                        # Heuristic: Time is likely the largest dimension
                                        time_dim = np.argmax(raw_out.shape)
                                        slicer = [slice(None)] * 3
                                        slicer[time_dim] = -1
                                        logic_outputs = raw_out[tuple(slicer)]

                                    # Plot individual truth table
                                    fig_tt = plot_truth_table(
                                        logic_inputs, 
                                        logic_outputs, 
                                        title=f"Logic Function Rank {rank+1} (Reward: {reward_val:.4f})",
                                        silent=True
                                    )
                                    
                                    # Construct unique name for logger
                                    plot_name = f'it {self.rendering_iteration} (Rank {rank+1})'

                                    if mode['format'] == 'figure':
                                        self.logger.log_figure(figure_name=plot_name, figure=fig_tt)
                                    elif mode['format'] == 'image':
                                        buf = BytesIO()
                                        fig_tt.savefig(buf, format='png')
                                        buf.seek(0)
                                        self.logger.log_image(buf, name=plot_name)
                                        buf.close()
                                    plt.close(fig_tt)
                        except Exception as e:
                            print(f"Could not plot truth tables: {e}")
                            pass

                    case {'style': 'logger', 'task': 'SSA_transients'}:
                        fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes, (list, np.ndarray)):
                            axes = [axes]
                        
                        # Overlay the SSA responses of the top k environments
                        for i in top_k:
                            # alpha=0.1 ensures the overlapping std dev shadings remain readable
                            fig, axes = self.envs[i].state.plot_SSA_transient_response(fig=fig, axes=axes, alpha=0.1)
                        
                        # fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN Distribution {self.rendering_iteration} (SSA)')
                        
                        # Apply bounds if provided
                        bounds = mode.get('bounds')
                        if bounds is not None:  
                            for a, b in zip(axes, bounds):
                                a.set_ylim([0, b])

                        # Logging
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration} SSA (Top {(1.-disregarded_percentage)*100}%)', figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN Distribution {self.rendering_iteration} SSA')
                            buf.close()
                        else:
                            raise ValueError(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)

                    case {'style': 'logger', 'task': 'habituation'}:
                        fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                        if not isinstance(axes, (list, np.ndarray)):
                            axes = [axes]
                        for i in top_k:
                            fig, axes = self.envs[i].state.plot_transient_response_piecewise(fig, axes)

                        # fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN Distribution {self.rendering_iteration}')
                        
                        # Apply bounds if provided
                        bounds = mode.get('bounds')
                        if bounds is not None:  
                            for a, b in zip(axes, bounds):
                                a.set_ylim([0, b])

                        # Logging
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration} (Top {(1.-disregarded_percentage)*100}%)', figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN Distribution {self.rendering_iteration}')
                            buf.close()
                        else:
                            raise ValueError(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)

                    # case {'style': 'logger', 'task': 'habituation_gap'}:
                    #     fig, axes = plt.subplots(self.envs[0].state.num_outputs, 1, figsize=(10, 5 * self.envs[0].state.num_outputs))
                    #     if not isinstance(axes, (list, np.ndarray)):
                    #         axes = [axes]
                    #     for i in top_k:
                    #         fig, axes = self.envs[i].state.plot_transient_response_piecewise(fig, axes, gap=True)

                    #     # fig.tight_layout(rect=[0, 0, 1, 0.95])
                    #     fig.suptitle(f'CRN Distribution {self.rendering_iteration}')
                        
                    #     # Apply bounds if provided
                    #     bounds = mode.get('bounds')
                    #     if bounds is not None:  
                    #         for a, b in zip(axes, bounds):
                    #             a.set_ylim([0, b])

                    #     # Logging
                    #     if mode['format'] == 'figure':
                    #         self.logger.log_figure(figure_name=f'CRN Distribution {self.rendering_iteration} (Top {(1.-disregarded_percentage)*100}%)', figure=fig)
                    #     elif mode['format'] == 'image':
                    #         buf = BytesIO()
                    #         fig.savefig(buf, format='png')
                    #         buf.seek(0)
                    #         self.logger.log_image(buf, name=f'CRN Distribution {self.rendering_iteration}')
                    #         buf.close()
                    #     else:
                    #         raise ValueError(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                    #     plt.close(fig)

                    
                    
        toc_step = time.time()
        if self.logger is not None:
            self.logger.log_metric('Timing: Render', toc_step - tic_step)