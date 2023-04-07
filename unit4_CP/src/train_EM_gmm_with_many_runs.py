'''
train_EM_gmm_with_many_runs.py

Use EM algorithm to train many GMMs, with different numbers of clusters and random seeds for initialization.

Post Condition
---------------
For each number of clusters K and seed, we will save:
* CSV file of the history for the GMM training process (perf metrics at every iteration)
* PNG file visualizing the final GMM model learned
within the results_many_EM_runs/ folder

'''

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from GMM_PenalizedMLEstimator_EM import GMM_PenalizedMLEstimator_EM
from viz_gmm_for_img_data import visualize_gmm

if __name__ == '__main__':
    dataset_name = 'tops-20x20flattened'
    x_train_ND = pd.read_csv("../data/%s_x_train.csv" % dataset_name).values
    x_valid_ND = pd.read_csv("../data/%s_x_valid.csv" % dataset_name).values
    # TODO load test data

    N, D = x_train_ND.shape
    K_list = [1, 4, 8, 16]
    seed_list = [1001, 3001, 4001, 7001]
    max_iter = 20

    # Create directory to store results, if it doesn't exist already
    results_dir = "results_many_EM_runs/"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    for K in K_list:
        best_seed = 0
        best_score = -1.0 * np.inf # init to lowest possible score

        for seed in seed_list:
            gmm_em = GMM_PenalizedMLEstimator_EM(
                K=K, D=D, seed=seed,
                max_iter=max_iter,
                variance_penalty_mode=25., variance_penalty_spread=100.0)

            print("Fitting with EM: K=%d seed=%d" % (K, seed))
            gmm_em.fit(x_train_ND, x_valid_ND)
            
            gmm_em.write_history_to_csv("results_many_EM_runs/history_K=%02d_seed=%04d.csv" % (K, seed))
            visualize_gmm(gmm_em, K)
            plt.savefig("results_many_EM_runs/viz_K=%02d_seed=%04d.png" % (K, seed),
                bbox_inches=0, pad_inches='tight')

            cur_score = gmm_em.history['valid_score_per_pixel'][-1]

        ## TODO determine which run was "best" in validation likelihood
        ## TODO assess best model on test data
        




