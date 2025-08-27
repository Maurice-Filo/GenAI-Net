import gymnasium as gym
from io import BytesIO
from matplotlib import pyplot as plt

class Environment(gym.Env):
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

    def reset(self):
        """
        Reset the state of the environment to an initial state by copying the CRN template
        """
        self.state = self.crn_template.clone()
        self.num_added_reactions = 0
        return self.state

    def step(self, action, mode='reaction index'):
        """
        Step through the environment with the given action.
        Args:
            action (int or float): The action to be taken, interpreted as either a rate constant or a complete reaction.
            mode (str): The mode of action interpretation, e.g., 'complex index', 'reaction index', or 'species index'.
        Returns:
            state (CRN object): The new state of the environment after taking the action.
            done (bool): A flag indicating whether the maximum number of reactions has been added.
        """
        # If the state (IOCRN) has unknown rate constants, the action is interpreted as a rate constant for the next unknown rate.
        # Otherwise the action is interpreted as a complete reaction.
        if self.state.num_unknown_rates > 0:
            self.state.set_next_unknown_rate(action)
        else:
            self.state.add_reaction(action, mode)
            self.num_added_reactions += 1  

        # Set a flag to indicate when the maximum number of reactions has been added        
        if self.num_added_reactions < self.max_added_reactions:
            done = False 
        else:
            done = True  
        return self.state, done
    
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
                - 'task': 'transients' for transient response, 'rank' for ranking.
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

            case {'style': 'logger', 'task': 'rank'}:
                if self.logger is not None:
                    self.logger.log_text(f"CRN {ID} \nRank={self.state.last_task_info['rank']}"+ str(self.state))
