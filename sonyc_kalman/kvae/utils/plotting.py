import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})

import pandas as pd

matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14


def plot_auxiliary(all_vars, filename, table_size=4):
    # All variables need to be (batch_size, sequence_length, dimension)
    for i, a in enumerate(all_vars):
        if a.ndim == 2:
            all_vars[i] = np.expand_dims(a, 0)

    dim = all_vars[0].shape[-1]
    if dim == 2:
        f, ax = plt.subplots(table_size, table_size, sharex='col', sharey='row', figsize=[12, 12])
        idx = 0
        for x in range(table_size):
            for y in range(table_size):
                for a in all_vars:
                    # Loop over the batch dimension
                    ax[x, y].plot(a[idx, :, 0], a[idx, :, 1], linestyle='-', marker='o', markersize=3)
                    # Plot starting point of the trajectory
                    ax[x, y].plot(a[idx, 0, 0], a[idx, 0, 1], 'r.', ms=12)
                idx += 1
        # plt.show()
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
        plt.close()
    else:
        df_list = []
        for i, a in enumerate(all_vars):
            df = pd.DataFrame(all_vars[i].reshape(-1, dim))
            df['class'] = i
            df_list.append(df)

        df_all = pd.concat(df_list)
        sns_plot = sns.pairplot(df_all, hue="class", vars=range(dim))
        sns_plot.savefig(filename)
    plt.close()


def plot_alpha(alpha, filename, idx=0):
    fig = plt.figure(figsize=[6, 6])
    ax = fig.gca()

    for line in np.swapaxes(alpha[idx], 1, 0):
        ax.plot(line, linestyle='-')

    ax.set_xlabel('Steps', fontsize=30)
    ax.set_ylabel('Mixture weight', fontsize=30)
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close()


def plot_alpha_grid(alpha, filename, table_size=4, idx=0):
    # All variables need to be (batch_size, sequence_length, dimension)
    if alpha.ndim == 2:
        alpha = np.expand_dims(alpha, 0)

    f, ax = plt.subplots(table_size, table_size, sharex='col', sharey='row', figsize=[12, 12])
    for x in range(table_size):
        for y in range(table_size):
            for i in range(alpha.shape[-1]):
                ax[x, y].plot(alpha[idx, :, i], linestyle='-', marker='o', markersize=3)
                ax[x, y].set_ylim([-0.01, 1.01])
            idx += 1
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close()


# JTC: We'll need to come up with some visualizations of our own here for the embeddings


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def plot_kalman_transfers(matrices, filename):
    fig, axarr = plt.subplots(1, len(matrices))
    for idx, mat in enumerate(matrices):
        hinton(mat, ax=axarr[idx])
    fig.savefig(filename, format='png', bbox_inches='tight', dpi=80)

