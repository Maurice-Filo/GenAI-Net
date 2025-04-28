from pathos.multiprocessing import ProcessingPool as Pool
import os
import time
from RL4CRN.Abstract.AbstractVecCRNEnvironment import AbstractCRNEnvironment

class VecEnv():
    def __init__(self, envs, N_CPUs=os.cpu_count(), logger=None):
        super().__init__(envs, logger=logger)
        self.N_CPUs = N_CPUs
        self.pool = Pool(N_CPUs)
    
    def get_reward(self, routine):
        tic_reward = time.time()
        rewards = self.pool.map(routine, [e.state for e in self.envs])
        toc_reward = time.time()
        if self.logger is not None:
            self.logger.log_metric('Reward Time', toc_reward - tic_reward)
        return rewards

    def close(self):
        self.pool.close()
        self.pool.join()

class SerialVecEnv(AbstractCRNEnvironment):
    def get_reward(self, routine):
        tic_reward = time.time()
        rewards = [routine(env.state) for env in self.envs]
        toc_reward = time.time()
        if self.logger is not None:
            self.logger.log_metric('Reward Time', toc_reward - tic_reward)
        return rewards