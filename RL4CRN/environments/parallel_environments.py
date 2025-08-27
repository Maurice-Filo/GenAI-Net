from pathos.multiprocessing import ProcessingPool as Pool
import os
import time
from RL4CRN.environments.abstract_multi_environments import AbstractMultiEnvironments

class ParallelEnvironments(AbstractMultiEnvironments):
    def __init__(self, envs, hall_of_fame_size, N_CPUs=os.cpu_count(), logger=None):
        """
        Initialize the parallel environments with a list of environments and the number of CPUs to use.
        Args:
            envs (list): A list of CRN environment instances.
            hall_of_fame_size (int): The size of the hall of fame to keep track of the best rewards.
            N_CPUs (int): The number of CPUs to use for parallel processing. Defaults to the number of available CPUs.
            logger (Logger, optional): An optional logger for logging metrics.
        """
        super().__init__(envs, hall_of_fame_size, logger=logger)
        self.N_CPUs = N_CPUs
        self.pool = Pool(N_CPUs)
    
    def get_reward(self, routine):
        """
        Get the reward from the routine based on the current state of all environments in parallel.
        Args:
            routine (function): A function that takes an environment state and returns a tuple of (reward, task_info).
        Returns:
            rewards_list: A list of rewards obtained from the routine for each environment.
            last_task_info_list: A list of information about the last task performed in each environment.
        """       
        tic_reward = time.time()
        results = self.pool.map(routine, [env.state for env in self.envs])
        rewards_list, last_task_info_list = zip(*results)
        rewards_list = list(rewards_list)
        last_task_info_list = list(last_task_info_list)

        # Update the last task info in each environment's state since running in parallel does not modify the environment state
        for i, env in enumerate(self.envs):
            env.state.last_task_info = last_task_info_list[i]
        toc_reward = time.time()
        if self.logger is not None:
            self.logger.log_metric('Reward Time', toc_reward - tic_reward)
        # Update the hall of fame
        if self.hall_of_fame_empty:
            combined_environments = self.envs
            self.hall_of_fame_empty = False
        else:
            combined_environments = self.hall_of_fame + self.envs
        sorted_environments = sorted(combined_environments, key=lambda x: x.state.last_task_info['reward'])
        for i in range(min(self.hall_of_fame_size, len(sorted_environments))):
            self.hall_of_fame[i].state = sorted_environments[i].state.clone()
        return rewards_list

    def close(self):
        """
        Close the parallel processing pool.
        This method should be called when the parallel environments are no longer needed to free up resources.
        """
        self.pool.close()
        self.pool.join()   