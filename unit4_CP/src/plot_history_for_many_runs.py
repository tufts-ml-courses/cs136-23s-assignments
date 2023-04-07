'''
plot_history_for_many_runs.py

Useful starter code for reading in CSV files and making plots

'''

import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.25)

ylabel_dict = {
    'valid_score_per_pixel':'val. score / pixel'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',
        default='results_many_EM_runs')
    parser.add_argument('--nickname',
        default='EM')
    parser.add_argument('--col_to_plot',
        default='valid_score_per_pixel')
    parser.add_argument('--y_lims_comma_sep_str',
        default="-0.4,0.9")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    nickname = args.nickname
    col_to_plot = args.col_to_plot
    y_lims = [float(y) for y in args.y_lims_comma_sep_str.split(",")]

    K_list = [1, 4, 8, 16]
    seed_list = [1001, 3001, 4001, 7001]
    fig, ax_grid = plt.subplots(
        nrows=1, ncols=len(K_list), 
        figsize=(12, 4),
        sharex=True, sharey=True, squeeze=False)
    for k, K in enumerate(K_list):
        for seed in seed_list:
            df = pd.read_csv(os.path.join(results_dir, 'history_K=%02d_seed=%04d.csv' % (K, seed)))
            ax_grid[0,k].plot(df['iter'], df[col_to_plot], '-', label='seed %d' % seed)

        ylabel = ylabel_dict.get(col_to_plot, col_to_plot)
        ax_grid[0,0].set_ylabel(ylabel)
        ax_grid[0,k].set_title("K=%d via %s" % (K, nickname))
        ax_grid[0,0].set_xticks([0, 5, 10, 15, 20])
        ax_grid[0,k].set_xlim([-1, 24])
        if y_lims is not None and len(y_lims) == 2:
            ax_grid[0,k].set_ylim(y_lims)
        ax_grid[0,k].set_xlabel('iter')
    ax_grid[0, -1].legend(loc='lower right', fontsize=10)
    plt.tight_layout();
    plt.show()