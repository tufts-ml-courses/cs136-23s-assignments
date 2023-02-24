import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from FeatureTransformPolynomial import PolynomialFeatureTransform

def make_fig_for_estimator(
        Estimator,
        order_list, alpha_list, beta_list,
        x_train_ND, t_train_N,
        test_scores_list=None,
        color='b',
        legend_label='MAP prediction',
        ):
    ''' Create figure showing estimator's predictions across orders

    Returns
    -------
    None

    Post Condition
    --------------
    Creates matplotlib figure
    '''
    fig, axgrid = prepare_x_vs_t_fig(order_list)
    xgrid_G1 = prepare_xgrid_G1(x_train_ND)

    # Loop over order of polynomial features
    # and associated axes of our plots
    for fig_col_id in range(len(order_list)):
        order = order_list[fig_col_id]
        alpha = alpha_list[fig_col_id]
        beta = beta_list[fig_col_id]
        cur_ax = axgrid[0, fig_col_id]

        feature_transformer = PolynomialFeatureTransform(
            order=order, input_dim=1)

        estimator = Estimator(
            feature_transformer, alpha=alpha, beta=beta)
        estimator.fit(x_train_ND, t_train_N)

        # Obtain predicted mean and stddev for estimator
        # at each x value in provided dense grid of size G
        mean_G = estimator.predict(xgrid_G1)
        var_G = estimator.predict_variance(xgrid_G1)

        plot_predicted_mean_with_filled_stddev_interval(
            cur_ax, # plot on figure's current axes
            xgrid_G1, mean_G, np.sqrt(var_G), # need square root of variance
            num_stddev=3,
            color=color,
            legend_label=legend_label)
    finalize_plot(
        axgrid, x_train_ND, t_train_N,
        order_list, alpha_list, beta_list)


def prepare_x_vs_t_fig(
        order_list,
        ):
    ''' Prepare figure for visualizing predictions on top of train data

    Returns
    -------
    fig : figure handle object
    axgrid : axis grid object
    '''
    nrows = 1
    ncols = len(order_list)
    panel_size = 3
    fig1, fig1_axgrid = plt.subplots(
        nrows=nrows, ncols=ncols,
        sharex=True, sharey=True, squeeze=False,
        figsize=(panel_size * ncols, panel_size * nrows))
    return fig1, fig1_axgrid


def prepare_xgrid_G1(
        x_train_ND,
        G=301,
        extrapolation_width_factor=0.5,
        ):
    '''
    
    Returns
    -------
    xgrid_G1 : 2D array, shape (G, 1)
        Grid of x points for making predictions 
    '''

    # To visualize prediction function learned from data,            
    # Create dense grid of G values between x.min() - R, x.max() + R
    # Basically, 2x as wide as the observed data values for 'x'
    xmin = x_train_ND[:,0].min()
    xmax = x_train_ND[:,0].max()
    R = extrapolation_width_factor * (xmax - xmin)
    xgrid_G = np.linspace(xmin - R, xmax + R, G)
    xgrid_G1 = np.reshape(xgrid_G, (G, 1))
    return xgrid_G1


def plot_predicted_mean_with_filled_stddev_interval(
        ax, xgrid_G1, t_mean_G, t_stddev_G,
        num_stddev=3,
        color='b',
        legend_label='MAP prediction',
        ):
    xgrid_G = np.squeeze(xgrid_G1)
    # Plot predicted mean and +/- 3 std dev interval
    ax.fill_between(
        xgrid_G,
        t_mean_G - num_stddev * t_stddev_G,
        t_mean_G + num_stddev * t_stddev_G,
        facecolor=color, alpha=0.2)
    ax.plot(
        xgrid_G, t_mean_G, 
        linestyle='-',
        color=color,
        label=legend_label)


def finalize_plot(
        axgrid, x_train_ND, t_train_N,
        order_list=None,
        alpha_list=None,
        beta_list=None):
    # Make figure beautiful
    for ii, ax in enumerate(axgrid.flatten()):
        transparency_level = 0.5
        ax.plot(x_train_ND, t_train_N,
            'k.', markersize=7, label='train data',
            alpha=transparency_level)
        alpha = alpha_list[ii]
        beta = beta_list[ii]
        order = order_list[ii]
        ax.set_title(
            "order = %d \n alpha=% .3g  beta=% .3g" % (
            order, alpha, beta))
        ax.set_ylim([-4, 4])
        ax.set_xlim([-4, 4])
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("input $x$")
        if ii == 0:
            ax.set_ylabel("predicted value $t$")
            ax.legend(loc='upper left')
