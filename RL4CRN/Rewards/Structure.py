import numpy as np

def compute_stoichiometry_rank(crn, target_rank):
    rank = np.linalg.matrix_rank(crn.stoichiometry_matrix)
    reward = np.abs(rank - target_rank)
    return reward,  {'task': 'rank', 'reward': reward, 'rank': rank, 'target_rank': target_rank}