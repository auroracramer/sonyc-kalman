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


def plot_segments(x_true_batch, x_hat_batch, mask_batch, a_batch, z_batch,
                  alpha_batch, filename, table_size=4, wh_ratio=2.5):
    batch_size, n_timesteps, x_dim = x_true_batch.shape
    a_dim = a_batch.shape[-1]
    z_dim = z_batch.shape[-1]

    x_aspect = (n_timesteps / x_dim) / wh_ratio
    a_aspect = (n_timesteps / a_dim) / wh_ratio
    z_aspect = (n_timesteps / z_dim) / wh_ratio

    x_vmin = min(x_true_batch.min(), x_hat_batch.min())
    x_vmax = max(x_true_batch.max(), x_hat_batch.max())

    a_vmin = a_batch.min()
    a_vmax = a_batch.max()

    z_vmin = z_batch.min()
    z_vmax = z_batch.max()


    table_size = min(table_size, batch_size)
    f, ax = plt.subplots(nrows=7, ncols=table_size, figsize=(25, 21))

    for idx in range(table_size):
        # Plot ground truth
        ax[0, idx].imshow(x_true_batch[idx].T, aspect=x_aspect,
                          cmap='magma', vmin=x_vmin, vmax=x_vmax)
        ax[0, idx].set_ylabel('Data dimension')
        ax[0, idx].set_title('Ground truth data ($x_t$)')

        # Make xticks consistent
        xticks = ax[0, idx].get_xticks()

        # Plot mask
        ax[1, idx].plot(mask_batch[idx])
        ax[1, idx].set_ylabel('Mask value')
        ax[1, idx].set_title('Imputation mask')
        ax[1, idx].set_ylim([0, 1.1])
        ax[1, idx].set_xticks(xticks)
        ax[1, idx].set_xlim([0, n_timesteps-1])

        # Plot reconstruction
        ax[2, idx].imshow(x_hat_batch[idx].T, aspect=x_aspect,
                          cmap='magma', vmin=x_vmin, vmax=x_vmax)
        ax[2, idx].set_ylabel('Feature dimension')
        ax[2, idx].set_title('Reconstructed data ($\hat{x}_t$)')
        ax[2, idx].set_xticks(xticks)
        ax[2, idx].set_xlim([0, n_timesteps-1])

        # Plot error
        ax[3, idx].plot(np.linalg.norm(x_true_batch[idx] - x_hat_batch[idx],
                                       axis=-1, ord=2) ** 2)
        ax[3, idx].set_ylabel('Squared L2 Error')
        ax[3, idx].set_title('L2 error between ground truth and reconstruction')
        ax[3, idx].set_xticks(xticks)
        ax[3, idx].set_xlim([0, n_timesteps-1])

        # Plot a_t
        ax[4, idx].imshow(a_batch[idx].T, aspect=a_aspect,
                          cmap='magma', vmin=a_vmin, vmax=a_vmax)
        ax[4, idx].set_ylabel('Feature dimension')
        ax[4, idx].set_title('Recognition latent variable ($a_t$)')
        ax[4, idx].set_xticks(xticks)
        ax[4, idx].set_xlim([0, n_timesteps-1])

        # Plot z_t
        ax[5, idx].imshow(z_batch[idx].T, aspect=z_aspect,
                          cmap='magma', vmin=z_vmin, vmax=z_vmax)
        ax[5, idx].set_ylabel('Feature dimension')
        ax[5, idx].set_title('Temporal latent mean ($E[z_t]$)')
        ax[5, idx].set_xticks(xticks)
        ax[5, idx].set_xlim([0, n_timesteps-1])

        # Plot alpha_t
        ax[6, idx].plot(alpha_batch[idx])
        ax[6, idx].set_xlabel('Steps')
        ax[6, idx].set_ylabel('Mixture weight')
        ax[6, idx].set_title('Mixture weights over time ($\\alpha^{(k)}_t$)')
        ax[6, idx].set_ylim([0, 1.1])
        ax[6, idx].set_xticks(xticks)
        ax[6, idx].set_xlim([0, n_timesteps-1])

    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
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
                ax[x, y].set_xlabel('Steps')
                ax[x, y].set_ylabel('Mixture weight')
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

