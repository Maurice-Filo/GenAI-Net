from io import BytesIO
from matplotlib import pyplot as plt
from RL4CRN.utils.visualizations import plot_truth_table
import numpy as np
from copy import deepcopy

class Environment():
    """
    Custom Environment that follows gym interface
    This is the basic environment for CRNs.
    """
    def __init__(self, crn_template, max_added_reactions, logger=None, logger_schedule=1):
        """
        Initialize the CRN environment with a template and maximum number of reactions.
        Args:
            crn_template (CRN object): The template for the CRNs to be generated.
            max_added_reactions (int): The maximum number of reactions allowed to be added to the CRN template.
            logger (Logger, optional): An optional logger for logging metrics.
            logger_schedule (int, optional): The frequency of logging updates.
        """
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
        """ Create a deep copy of the environment. """
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
        """ Reset the state of the environment to an initial state by copying the CRN template. """
        self.state = self.crn_template.clone()
        self.num_added_reactions = 0
        self.actions_taken = []
        self.raw_actions_taken = []
        return self.state

    def step(self, action, stepper, raw_action=None):

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
        """
        Get the action taken at a specific index.
        Args:
            index (int): The index of the action to retrieve.
        Returns:
            action: The action taken at the specified index.
        """
        return self.actions_taken[index]
    
    def get_raw_action(self, index):
        """
        Get the raw action taken at a specific index.
        Args:
            index (int): The index of the raw action to retrieve.
        Returns:
            raw_action: The raw action taken at the specified index.
        """
        return self.raw_actions_taken[index]
    
    def get_reward(self, routine):
        """
        Get the reward from the routine based on the current state of the environment.
        Args:
            routine (function): A function that takes the current state and returns a tuple (reward, last_task_info).
        Returns:
            rewards (float): The reward obtained from the routine.
            last_task_info (dict): Information about the last task performed.
        """
        rewards, last_task_info = routine(self.state)[0]
        return rewards

    def render(self, mode={'style': 'human', 'task': 'transients', 'format': 'figure'}, ID=None):
        """
        Render the current state of the environment.
        Args:
            mode (dict): A dictionary specifying the rendering style and task.
                - 'style': 'human' for interactive display, 'logger' for logging to a file.
                - 'task': 'transients' for transient response, 'rank' for matrix rank.
                - 'format': 'figure' or 'image' for the output format.
            ID (str, optional): An identifier for the CRN, used in logging.
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