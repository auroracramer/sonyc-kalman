import json
import os

import numpy as np
import tensorflow as tf
from kvae import KalmanVariationalAutoencoder
from kvae.utils import reload_config, get_train_config

from data import construct_kvae_data

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})

np.random.seed(1337)


def run():
    """Load and train model

    Create a model object and run the training using the provided config.
    """
    # JTC: We'll have to change this script
    config = get_train_config()
    # To reload a saved model
    config = reload_config(config.FLAGS)

    # Load data:
    if not config.data_path:
        raise ValueError('Must provide path to data.')

    if not os.path.isfile(config.data_path):
        err_msg = 'Invalid path to data: {}'
        raise ValueError(err_msg.format(config.data_path))

    if not config.train_mask_path:
        raise ValueError('Must provide path to training mask.')

    if not os.path.isfile(config.train_mask_path):
        err_msg = 'Invalid path to training mask: {}'
        raise ValueError(err_msg.format(config.train_mask_path))

    if not config.test_mask_path:
        raise ValueError('Must provide path to testing mask.')

    if not os.path.isfile(config.test_mask_path):
        err_msg = 'Invalid path to testing mask: {}'
        raise ValueError(err_msg.format(config.test_mask_path))

    data = np.load(config.data_path)
    X, mask = data['X'], data['mask']
    train_subset_mask = np.load(config.train_mask_path)
    test_subset_mask = np.load(config.test_mask_path)

    train_data, train_mask = construct_kvae_data(X, mask, train_subset_mask,
                                                 n_timesteps=config.n_timesteps,
                                                 hop_length=config.hop_length)
    test_data, test_mask = construct_kvae_data(X, mask, test_subset_mask, test=True)

    # Add timestamp to log path
    config.log_dir = os.path.join(config.log_dir, '%s' % config.run_name)

    # Add model name to log path
    config.log_dir = config.log_dir + '_kvae'

    # Create log path
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)

    # Save hyperparameters
    with open(config.log_dir + '/config.json', 'w') as f:
        json.dump(dict(config.__flags), f)


    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    with tf.Session() as sess:
        model = KalmanVariationalAutoencoder(train_data, test_data, train_mask, test_mask, config, sess)

        model.build_model().build_loss().initialize_variables()
        err = model.train()
        model.imputation_plot('missing_planning')
        model.imputation_plot('missing_random')

        # # Plots only, remember to load a pretrained model setting reload_model
        # # to e.g. logdir/20170322110525/model.ckpt
        # model.build_model().initialize_variables().impute()
        # model.generate()

        return err

if __name__ == "__main__":
    lb = run()
