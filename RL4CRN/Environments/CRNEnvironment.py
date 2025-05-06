import gymnasium as gym
from copy import deepcopy
import numpy as np
from io import BytesIO
from matplotlib import pyplot as plt

class CRNEnvironment(gym.Env):
    """
    Custom Environment that follows gym interface
    This is the basic environment for CRNs.
    """
    def __init__(self, CRN_template, max_num_reactions, logger=None, logger_schedule=1):
        super(CRNEnvironment, self).__init__()
        self.CRN_template = CRN_template
        self.action_space = gym.spaces.Dict({
            'reactants space': gym.spaces.Discrete(self.CRN_template.get_complexes_range()),
            'products space': gym.spaces.Discrete(self.CRN_template.get_complexes_range()),
            'input influence space': gym.spaces.Discrete(self.CRN_template.num_inputs + 1),
            'rate constant space': gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.CRN_template.num_species,), dtype=np.float32)
        self.state = deepcopy(self.CRN_template)
        self.num_added_reactions = 0
        self.max_num_reactions = max_num_reactions
        self.logger = logger
        self.logger_schedule = logger_schedule

    def reset(self):
        """
        Reset the state of the environment to an initial state by copying the CRN template
        """
        self.state = deepcopy(self.CRN_template)
        self.num_added_reactions = 0
        return self.state

    def step(self, action):
        """
        Act on the CRN environment by:
        1- filling the next unknown parameter if there are any
        2- adding a reaction to the CRN otherwise.
        For the first case, the action is a float, and for the second case, it is a dictionary of the following format
        {
            'reactants space': gym.spaces.Discrete(self.CRN_template.get_complexes_range()),
            'products space': gym.spaces.Discrete(self.CRN_template.get_complexes_range()),
            'input influence space': gym.spaces.Discrete(self.CRN_template.num_inputs + 1),
            'rate constant space': gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        }
        """
        if self.state.num_unknown_parameters > 0:
            self.state.set_next_unknown_parameter(action)
        else:
            self.state.add_reaction(action)
            self.num_added_reactions += 1      
        if self.num_added_reactions < self.max_num_reactions:
            done = False 
        else:
            done = True  
        info = {} 
        return self.state, done, info
    
    def get_reward(self, routine):
        rewards = routine(self.state)[0]  # get the reward from the routine which returns a tuple (reward, last_task_info)
        return rewards

    def render(self, mode={'style': 'human', 'task': 'transients', 'format': 'figure'}, ID=None):
        match mode:
            case {'style': 'human'}:
                self.state.plot_transient_response()
            case {'style': 'logger', 'task': 'transients'}:
                if self.logger is not None:
                    self.logger.log_text(f'CRN {ID} \n' + str(self.state))
                    try:
                        fig, _ = self.state.plot_transient_response()
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.suptitle(f'CRN {ID}')
                        if mode['format'] == 'figure':
                            self.logger.log_figure(figure_name=f'CRN {ID}', figure=fig)
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
