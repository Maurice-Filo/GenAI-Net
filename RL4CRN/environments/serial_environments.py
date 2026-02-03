"""
Serial multi-environment evaluation.

This module defines `SerialEnvironments`, a subclass of
`RL4CRN.environments.abstract_multi_environments.AbstractMultiEnvironments`
that evaluates rewards for multiple environments sequentially (no parallelism).

This is useful for:

- debugging and deterministic profiling,
- small batch sizes where multiprocessing overhead dominates.
"""

import time
import bisect
from RL4CRN.environments.abstract_multi_environments import AbstractMultiEnvironments

class SerialEnvironments(AbstractMultiEnvironments):
    """Multi-environment manager with serial reward evaluation."""

    def __init__(self, envs, hall_of_fame_size, logger=None):
        """Initialize the serial environments wrapper.

        Args:
            envs: List of CRN environment instances.
            hall_of_fame_size: Maximum number of environments stored in the hall
                of fame (0 disables hall of fame).
            logger: Optional logger for metrics.
        """
        super().__init__(envs, hall_of_fame_size, logger=logger)

    def get_reward(self, routine):
        """Evaluate rewards for all environments sequentially.

        The provided `routine` is applied to each environment's state in a Python
        loop. The routine is expected to return a tuple `(reward, task_info)` for
        each state.

        Note:
            Unlike `RL4CRN.environments.parallel_environments.ParallelEnvironments`,
            this method does not explicitly write `task_info` back into
            `env.state.last_task_info` as it is expected to be handled by the routine itself (default).

        Args:
            routine: Callable taking an environment state and returning
                `(reward, task_info)`.

        Returns:
            Sequence of reward values, one per environment.

        Side Effects:
            - Logs `'Reward Time'` if a logger is available.
            - Adds all environments to the hall of fame (if enabled).
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