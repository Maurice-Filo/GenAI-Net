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
        rewards_list, last_task_info_list = [routine(env.state) for env in self.envs]
        toc_reward = time.time()
        if self.logger is not None:
            self.logger.log_metric('Reward Time', toc_reward - tic_reward)

        # Update the hall of fame
        indices = top_k_smallest_indices_sorted(rewards_list, self.hall_of_fame_size)
        if len(self.hall_of_fame) == 0:
            self.hall_of_fame = [
                {'environment': self.envs[i], 'reward': rewards_list[i]}
                for i in indices
            ]
        else:
            current_rewards = [entry['reward'] for entry in self.hall_of_fame]
            for i in indices:
                reward = rewards_list[i]
                if reward >= current_rewards[-1]:
                    break
                else:
                    env = self.envs[i]
                    insert_index = bisect.bisect_left(current_rewards, reward)
                    self.hall_of_fame.insert(insert_index, {'environment': env, 'reward': reward})
                    current_rewards.insert(insert_index, reward)
                    self.hall_of_fame.pop()
                    current_rewards.pop()

        return rewards_list