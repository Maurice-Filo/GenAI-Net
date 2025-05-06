from pathos.multiprocessing import ProcessingPool as Pool
import os
import time
from RL4CRN.Environments.AbstractVecCRNEnvironment import AbstractVecCRNEnvironment

class VecCRNEnvironment(AbstractVecCRNEnvironment):
    def __init__(self, envs, N_CPUs=os.cpu_count(), logger=None):
        super().__init__(envs, logger=logger)
        self.N_CPUs = N_CPUs
        self.pool = Pool(N_CPUs)
    
    def get_reward(self, routine):
        tic_reward = time.time()
        outputs = self.pool.map(routine, [e.state for e in self.envs])
        for i, env in enumerate(self.envs):
            env.state.last_task_info = outputs[i][1]
            # for attr, value in outputs[i][1].items():
            #     setattr(env.state, attr, value)
            # env.state.last_trajectories = outputs[i][1]
            # env.state.last_time_horizon = outputs[i][2]
        rewards = [output[0] for output in outputs]
        toc_reward = time.time()
        if self.logger is not None:
            self.logger.log_metric('Reward Time', toc_reward - tic_reward)
        return rewards

    def close(self):
        self.pool.close()
        self.pool.join()

class SerialVecCRNEnvironment(AbstractVecCRNEnvironment):
    def get_reward(self, routine):
        tic_reward = time.time()
        rewards = [routine(env.state)[0] for env in self.envs]
        toc_reward = time.time()
        if self.logger is not None:
            self.logger.log_metric('Reward Time', toc_reward - tic_reward)
        return rewards