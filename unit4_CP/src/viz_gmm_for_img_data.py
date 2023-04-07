import numpy as np
import matplotlib.pyplot as plt


def visualize_gmm(gmm, max_K_to_display=8, fontsize=18):
    ''' Create single image visualization of all GMM parameters

    Post Condition
    --------------
    New matplotlib figure created with visual of means and stddevs for all K clusters.
    '''
    K = gmm.K
    D = gmm.D
    P = int(np.sqrt(D))

    comp_ids_bigtosmall_K = np.argsort(gmm.log_pi_K)[::-1][:max_K_to_display]

    ncols = max_K_to_display + 1
    fig, ax_grid = plt.subplots(
        nrows=2, ncols=ncols,
        figsize=(3 * ncols, 3 * 2),
        squeeze=False)

    ax_grid[0,0].set_ylabel("mean", fontsize=fontsize)
    ax_grid[1,0].set_ylabel("stddev", fontsize=fontsize)
    last_col_id = comp_ids_bigtosmall_K.size - 1
    for col_id, kk in enumerate(comp_ids_bigtosmall_K):
        # Plot learned means
        cur_ax = ax_grid[0, col_id]
        mu_img_PP = gmm.mu_KD[kk].reshape((P, P))
        img_h = cur_ax.imshow(mu_img_PP, interpolation='nearest',
            vmin=-1.0, vmax=1.0, cmap='gray')
        cur_ax.set_title("k = %d  %4.1f%%" % (kk, 
            100*np.exp(gmm.log_pi_K[kk])), fontsize=fontsize)
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        if col_id == last_col_id:
            cbar = fig.colorbar(img_h, ax=ax_grid[0, col_id + 1],
                location='left', ticks=[-1.0, 0.0, 1.0])
            cbar.ax.tick_params(labelsize=fontsize)

        # Plot learned stddev
        cur_ax = ax_grid[1, col_id]
        stddev_img_PP = gmm.stddev_KD[kk].reshape((P, P))
        img_h  = cur_ax.imshow(stddev_img_PP, interpolation='nearest',
            vmin=0.0, vmax=1.5, cmap='afmhot')
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        if col_id == last_col_id:
            cbar = fig.colorbar(img_h, ax=ax_grid[1, col_id + 1],
                location='left', ticks=[0.0, 0.5, 1.0])
            cbar.ax.tick_params(labelsize=fontsize)

    for empty_kk in range(K, ncols):
        empty_ax=ax_grid[0, empty_kk]
        empty_ax.set_visible(False)
        empty_ax=ax_grid[1, empty_kk]
        empty_ax.set_visible(False)
    plt.tight_layout()
