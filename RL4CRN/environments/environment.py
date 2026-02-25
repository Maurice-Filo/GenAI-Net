"""
Single-environment wrapper for CRN design via reinforcement learning.

This module defines `Environment`, a lightweight environment wrapper with
a Gym-like interface tailored to chemical reaction networks (CRNs). The
environment maintains a mutable CRN state initialized from a template and allows
an agent to *add reactions* up to a fixed budget.

Core loop:
    - `reset` clones the CRN template into the current state.
    - `step` applies an action to the current state via a provided
      *stepper* and increments the reaction budget counter.
    - The environment returns `(state, done)` where `done` indicates whether the
      maximum number of added reactions has been reached.

Action semantics:
    The environment itself does not interpret actions. Instead, it delegates
    state updates to a `stepper` object with a `step(state, action)` method
    (see `RL4CRN.agent2env_interface.abstract_stepper.AbstractStepper`).

Logging and rendering:
    `render` supports a number of plotting/logging tasks driven by a
    `mode` dictionary. In `'logger'` mode, plots are logged via the provided
    logger as either figures or PNG images.
"""

from io import BytesIO
from matplotlib import pyplot as plt
from RL4CRN.utils.visualizations import plot_truth_table
import numpy as np
from copy import deepcopy

class Environment():
    """Gym-like CRN environment based on adding reactions to a template.

    Args:
        crn_template: CRN object used as the initial template. Must provide
            `clone()` and should expose plotting methods used by `render`
            (e.g., `plot_transient_response`, `plot_phase_portrait`, etc.).
        max_added_reactions: Maximum number of reactions that can be added before
            the environment signals termination (`done=True`).
        logger: Optional logger used by `render` in `'logger'` mode.
            Expected to provide methods such as `log_text`, `log_figure`, and
            `log_image`.
        logger_schedule: Frequency of logging updates (stored for downstream use;
            not actively enforced in the current implementation).

    Attributes:
        state: Current CRN state (a clone of `crn_template`, then mutated).
        num_added_reactions: Number of actions applied since last reset.
        actions_taken: List of environment actions applied via `step`.
        raw_actions_taken: Optional list of raw policy actions (if provided to
            `step`).
    """

    def __init__(self, crn_template, max_added_reactions, logger=None, logger_schedule=1):
        super(Environment, self).__init__()
        self.crn_template = crn_template
        self.state = self.crn_template.clone()
        self.num_added_reactions = 0
        self.max_added_reactions = max_added_reactions
        self.logger = logger
        self.logger_schedule = logger_schedule
        self.actions_taken = []
        self.raw_actions_taken = []

    def clone(self):
        """Create a deep copy of the environment.

        The clone includes:
            - a clone of the template and current state,
            - the current reaction count,
            - copies of `actions_taken` and `raw_actions_taken`.

        Returns:
            A new `Environment` instance with duplicated internal state.
        """
        base = Environment(
            crn_template=self.crn_template.clone(),
            max_added_reactions=self.max_added_reactions,
            logger=self.logger,
            logger_schedule=self.logger_schedule
        )
        base.state = self.state.clone()
        base.num_added_reactions = self.num_added_reactions
        base.actions_taken = deepcopy(self.actions_taken)
        base.raw_actions_taken = deepcopy(self.raw_actions_taken)
        return base

    def reset(self):
        """Reset the environment state to the template.

        This clears the reaction counter and stored action histories.

        Returns:
            The reset CRN state (clone of the template).
        """
        self.state = self.crn_template.clone()
        self.num_added_reactions = 0
        self.actions_taken = []
        self.raw_actions_taken = []
        return self.state

    def step(self, action, stepper, raw_action=None):
        """Apply an action to the CRN state via a stepper.

        The `stepper` is responsible for mutating the current state given the
        action. After applying the action, the environment increments
        `num_added_reactions` and returns a termination flag indicating whether
        the reaction budget is exhausted.

        Args:
            action: Environment action to apply (typically a reaction or a
                reaction-like object).
            stepper: Object providing `step(state, action)` that mutates the
                state in-place.
            raw_action: Optional raw policy action (stored in `raw_actions_taken`
                if provided). Useful for algorithms that require access to the
                policy outputs (e.g., self-imitation learning).

        Returns:
            Tuple `(state, done)` where:
                - `state` is the updated CRN state.
                - `done` is True if `num_added_reactions >= max_added_reactions`.
        """

        stepper.step(self.state, action)
        self.num_added_reactions += 1  

        # Set a flag to indicate when the maximum number of reactions has been added        
        if self.num_added_reactions < self.max_added_reactions:
            done = False 
        else:
            done = True  

        # Store the action for algorithms that may need it
        
        self.actions_taken.append(action)
        if raw_action is not None:
            self.raw_actions_taken.append(raw_action)

        return self.state, done
    
    def get_action(self, index):
        """Return the environment action taken at a given step index.

        Args:
            index: Index into `actions_taken`.

        Returns:
            The action stored at the specified index.

        Raises:
            IndexError: If `index` is out of range.
        """
        return self.actions_taken[index]
    
    def get_raw_action(self, index):
        """Return the raw policy action stored at a given step index.

        Args:
            index: Index into `raw_actions_taken`.

        Returns:
            The raw action stored at the specified index.

        Raises:
            IndexError: If `index` is out of range.
        """
        return self.raw_actions_taken[index]
    
    def get_reward(self, routine):
        """Compute a reward (or loss) for the current CRN state.

        The `routine` is expected to evaluate the current state and return a
        structure whose first element is a tuple `(reward, last_task_info)`.
        This method extracts the scalar reward component and returns it.

        Args:
            routine: Callable taking the current state and returning an iterable
                whose first element is `(reward, last_task_info)`.

        Returns:
            Scalar reward value (as produced by `routine`).

        Notes:
            The `last_task_info` returned by the routine is not propagated by this
            method. In most workflows it is expected to be stored inside
            `self.state.last_task_info` by the routine/state implementation.
        """
        rewards, last_task_info = routine(self.state)[0]
        return rewards

    def render(self, mode={'style': 'human', 'task': 'transients', 'format': 'figure'}, ID=None):
        """Render or log diagnostics for the current CRN state.

        Rendering behavior is controlled by a `mode` dictionary. Supported cases
        include interactive plotting (`'human'`) and logging plots to a logger
        (`'logger'`).

        Args:
            mode: Dictionary describing rendering behavior. Typical keys:
            
                - `style`: `'human'` or `'logger'`.
                - `task`: Diagnostic task. Supported values in this implementation
                  include `'transients'`, `'phase_plot'`, `'rank'`,
                  `'transients + dose-response'`, `'transients + frequency content'`,
                  `'transients + logic'`, `'SSA_transients'`.
                - `format`: `'figure'` to log matplotlib figures directly, or
                  `'image'` to log PNG buffers.
                Some tasks also consume optional keys such as `t0`, `bounds_freq`,
                or `scale`.
            ID: Optional identifier used when naming logged artifacts.

        Returns:
            None.

        Side Effects:
            - In `'human'` mode, opens matplotlib windows (depending on backend).
            - In `'logger'` mode, logs text/figures/images via `self.logger`.
            - Creates and closes matplotlib figures.
        """
        match mode:
            case {'style': 'human'}:
                self.state.plot_transient_response()
                
            case {'style': 'logger', 'task': 'transients'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    try:
                        fig, _ = self.state.plot_transient_response()
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID}, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID}", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID}')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    except ValueError:
                        print("Warning: Could not plot transient response.")
                        pass

            case {'style': 'logger', 'task': 'phase_plot'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    try:
                        fig, _ = self.state.plot_transient_response(alpha=1.0)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID}, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID}", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID}')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    except ValueError:
                        pass

                    try:
                        fig1, _ = self.state.plot_phase_portrait(alpha=1.0)
                        fig1.tight_layout(rect=[0, 0, 1, 0.95])
                        fig1.suptitle(f"CRN {ID}, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID} Phase Portrait", figure=fig1)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig1.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID} Phase Portrait')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig1)
                    except ValueError:
                        pass

            case {'style': 'logger', 'task': 'rank'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID} \nRank={self.state.last_task_info['rank']}"+ str(self.state))

            case {'style': 'logger', 'task': 'transients + dose-response'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    try:
                        fig, _ = self.state.plot_transient_response()
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID}, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID}", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID}')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    except ValueError:
                        pass

                    try:
                        fig1, _ = self.state.plot_dose_response()
                        fig1.tight_layout(rect=[0, 0, 1, 0.95])
                        fig1.suptitle(f"CRN {ID}, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID} Dose Response", figure=fig1)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig1.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID} Dose Response')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig1)
                    except ValueError:
                        pass

            case {'style': 'logger', 'task': 'transients + frequency content'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    try:
                        fig, _ = self.state.plot_transient_response(alpha=1.0)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID}, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID}", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID}')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    except ValueError:
                        pass

                    try:
                        fig1, axes1 = self.state.plot_frequency_content(alpha=1.0, t0=mode.get('t0', 0.0))
                        fig1.tight_layout(rect=[0, 0, 1, 0.95])
                        fig1.suptitle(f"CRN {ID}, Reward: {self.state.last_task_info['reward']}")
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
                            self.logger.log_figure(figure_name=f"CRN {ID} Frequency Content", figure=fig1)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig1.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID} Frequency Content')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig1)
                    except ValueError:
                        pass

            case {'style': 'logger', 'task': 'transients + logic'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    
                    # 1. Plot Transients
                    try:
                        fig, _ = self.state.plot_transient_response(alpha=1.0)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID} Transients, Reward: {self.state.last_task_info['reward']}")
                        
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID} Transients", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID} Transients')
                            buf.close()
                        plt.close(fig)
                    except ValueError:
                        pass

                    # 2. Plot Logic Truth Table
                    logic_inputs = self.state.last_task_info.get('inputs')
                    logic_outputs = self.state.last_task_info.get('outputs') 
                    
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

                        fig_tt = plot_truth_table(
                            logic_inputs, 
                            logic_outputs, 
                            title=f"{ID} Truth Table",
                            silent=True
                        )
                        
                        # Construct unique name for logger
                        plot_name = f"{ID} Truth Table"

                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=plot_name, figure=fig_tt)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig_tt.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=plot_name)
                            buf.close()
                        plt.close(fig_tt)

            case {'style': 'logger', 'task': 'SSA_transients'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    try:
                        # alpha=0.2 ensures the std dev shading is transparent
                        fig, _ = self.state.plot_SSA_transient_response(alpha=0.2)
                        # fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID} (SSA), Reward: {self.state.last_task_info['reward']}")
                        
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID} SSA", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID} SSA')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    except ValueError:
                        pass

            case {'style': 'logger', 'task': 'habituation'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    try:
                        fig, _ = self.state.plot_transient_response_piecewise()
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID} Habituation Response, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID} Habituation", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID} Habituation')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    except ValueError:
                        pass

            case {'style': 'logger', 'task': 'habituation_gap'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    try:
                        fig, _ = self.state.plot_transient_response_piecewise(gap=True)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID} Habituation Response, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID} Habituation", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID} Habituation')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    except ValueError:
                        pass

            case {'style': 'logger', 'task': 'sensitization_gap'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID}, Reward: {self.state.last_task_info['reward']} \n" + str(self.state))
                    try:
                        fig, _ = self.state.plot_transient_response_piecewise(gap=True)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f"CRN {ID} Sensitization Response, Reward: {self.state.last_task_info['reward']}")
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f"CRN {ID} Sensitization", figure=fig)
                        elif mode['format'] == 'image':
                            buf = BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            self.logger.log_image(buf, name=f'CRN {ID} Sensitization')
                            buf.close()
                        else:
                            raise Exception(f"Unknown mode: {mode['format']}. Use 'figure' or 'image'.")
                        plt.close(fig)
                    except ValueError:
                        pass