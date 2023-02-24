'''
Summary
-------
Plot predicted mean + high confidence interval for MAP estimator
across different orders of the polynomial features
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")

import regr_viz_utils
from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator
from LinearRegressionPosteriorPredictiveEstimator import LinearRegressionPosteriorPredictiveEstimator


if __name__ == '__main__':    

    order_list = [1, 3, 9]

    # Set precisions of prior (alpha) and likelihood (beta)
    alpha_list = 0.01 * np.ones(3)
    beta_list = 4.0 * np.ones(3) 

    # Set training set size
    N = 8 

    # Load and unpack training and test data
    data_dir = os.path.abspath('../data/')
    train_csv_fpath = os.path.join(data_dir, 'toywave_train.csv')
    test_csv_fpath = os.path.join(data_dir, 'toywave_test.csv')
    train_df = pd.read_csv(train_csv_fpath)
    test_df = pd.read_csv(test_csv_fpath)
    x_train_ND = train_df['x'].values[:,np.newaxis]
    t_train_N = train_df['y'].values
    x_test_ND = test_df['x'].values[:,np.newaxis]
    t_test_N = test_df['y'].values

    map_fig, map_axgrid = regr_viz_utils.prepare_x_vs_t_fig(order_list)
    xgrid_G1 = regr_viz_utils.prepare_xgrid_G1(x_train_ND)

    # Loop over order of polynomial features
    for fig_col_id in range(len(order_list)):
        order = order_list[fig_col_id]
        alpha = alpha_list[fig_col_id]
        beta = beta_list[fig_col_id]
        cur_map_ax = map_axgrid[0, fig_col_id]

        feature_transformer = PolynomialFeatureTransform(
            order=order, input_dim=1)

        # Train MAP estimator using only first N examples
        map_estimator = LinearRegressionMAPEstimator(
            feature_transformer, alpha=alpha, beta=beta)
        map_estimator.fit(x_train_ND[:N], t_train_N[:N])

        # Obtain predicted mean and stddev for MAP estimator
        # at each x value in provided dense grid of size G
        map_mean_G = map_estimator.predict(xgrid_G1)
        map_stddev_G = np.ones(map_mean_G.size) # TODO FIXME predict_variance

        regr_viz_utils.plot_predicted_mean_with_filled_stddev_interval(
            cur_map_ax, # plot on MAP figure's current axes
            xgrid_G1, map_mean_G, map_stddev_G,
            num_stddev=3,
            color='b',
            legend_label='MAP prediction')

    regr_viz_utils.finalize_plot(
        map_axgrid, x_train_ND[:N], t_train_N[:N],
        order_list, alpha_list, beta_list)
    plt.show()
