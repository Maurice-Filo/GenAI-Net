import time
import bisect
from RL4CRN.environments.abstract_multi_environments import AbstractMultiEnvironments

class SerialEnvironments(AbstractMultiEnvironments):
    def __init__(self, envs, hall_of_fame_size, logger=None):
        """
        Initialize the serilal environments with a list of environments.
        Args:
            envs (list): A list of CRN environment instances.
            hall_of_fame_size (int): The size of the hall of fame to keep track of the best rewards.
            logger (Logger, optional): An optional logger for logging metrics.
        """
        super().__init__(envs, hall_of_fame_size, logger=logger)

    def get_reward(self, routine):
        """
        Get the reward from the routine based on the current state of all environments in serial.
        Args:
            routine (function): A function that takes an environment state and returns a reward.
        Returns:
            rewards_list: A list of rewards obtained from the routine for each environment.
            last_task_info_list: A list of information about the last task performed in each environment.
        """
        tic_reward = time.time()
        rewards_list, last_task_info_list = zip(*[routine(env.state) for env in self.envs])
        toc_reward = time.time()
        if self.logger is not None:
            self.logger.log_metric('Reward Time', toc_reward - tic_reward)

        # Update the hall of fame
        if self.hall_of_fame is not None:
            self.hall_of_fame.add_all(self.envs)
        
        return rewards_list