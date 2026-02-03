"""
Parallel multi-environment evaluation.

This module defines `ParallelEnvironments`, a subclass of
`RL4CRN.environments.abstract_multi_environments.AbstractMultiEnvironments`
that evaluates rewards for multiple environments in parallel using
`joblib`.

Only the reward computation is parallelized here (via `get_reward`).
Stepping and observation remain synchronous and are inherited from the base
class.
"""

# from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed
import os
import time
from RL4CRN.environments.abstract_multi_environments import AbstractMultiEnvironments

class ParallelEnvironments(AbstractMultiEnvironments):
    """Multi-environment manager with parallel reward evaluation.

    Args:
        envs: List of CRN environment instances.
        hall_of_fame_size: Maximum number of environments stored in the hall of
            fame (0 disables hall of fame).
        N_CPUs: Number of worker processes for parallel computation. Defaults to
            `os.cpu_count()`.
        logger: Optional logger for metrics.

    Attributes:
        N_CPUs: Number of parallel workers used for reward computation.
    """

    def __init__(self, envs, hall_of_fame_size, N_CPUs=os.cpu_count(), logger=None):
        super().__init__(envs, hall_of_fame_size, logger=logger)
        self.N_CPUs = N_CPUs
    
    def get_reward(self, routine):
        """Evaluate rewards for all environments in parallel.

        The provided `routine` is applied to each environment's state using
        `joblib.Parallel`. The routine is expected to return a tuple
        `(reward, task_info)` for a given state.

        Because evaluation happens outside the environment objects, this method
        explicitly writes `task_info` back into each `env.state.last_task_info`.

        Args:
            routine: Callable taking an environment state and returning
                `(reward, task_info)`.

        Returns:
            List of reward values, one per environment.

        Side Effects:
            - Sets `env.state.last_task_info` for each environment.
            - Logs `'Timing: Rewards'` if a logger is available.
            - Adds all environments to the hall of fame (if enabled).
        """    
        tic_reward = time.time()
        results = Parallel(n_jobs=self.N_CPUs)(delayed(routine)(env.state) for env in self.envs)
        # results = self.pool.map(routine, [env.state for env in self.envs])
        rewards_list, last_task_info_list = zip(*results)
        rewards_list = list(rewards_list)
        last_task_info_list = list(last_task_info_list)

        # Update the last task info in each environment's state since running in parallel does not modify the environment state
        for i, env in enumerate(self.envs):
            env.state.last_task_info = last_task_info_list[i]
        toc_reward = time.time()
        if self.logger is not None:
            self.logger.log_metric('Timing: Rewards', toc_reward - tic_reward)
            
        # update hall of fame 
        if self.hall_of_fame is not None:
            self.hall_of_fame.add_all(self.envs)

        return rewards_list
