import argparse
import datetime
import os
import pandas as pd
import h5py
import numpy as np

from . import data_aggr


def load_openl3_time_series(hdf5_path, delta_mins=15, aggr_func=None):
    if aggr_func is None:
        # By default, just take the center embedding
        aggr_func = lambda x: x[len(x)//2]

    # Set slot duration
    delta = datetime.timedelta(minutes=delta_mins)

    with h5py.File(hdf5_path) as f:

        # Get embedding size
        emb_size = f['openl3'][0]['openl3'].shape[1]

        # Initialize
        X = []
        mask = []
        curr_slot_emb_list = []
        curr_upper_ts = datetime.datetime.utcfromtimestamp(f['openl3'][0]['timestamp']) + delta

        # Construct time series
        for idx in range(len(f['openl3'])):
            # Get the timestamp for the current step
            ts = datetime.datetime.utcfromtimestamp(f['openl3'][idx]['timestamp'])

            if ts < curr_upper_ts:
                # If in the current slot, add to the list of embeddings for this slot
                curr_slot_emb_list += list(f['openl3'][idx]['openl3'])
            else:
                # If not in the current slot, aggregate embeddings for current slot
                if curr_slot_emb_list:
                    slot_emb = aggr_func(curr_slot_emb_list)
                    X.append(slot_emb)
                    mask.append(0)
                else:
                    # If there are no embeddings (i.e. there is missing data), add a dummy
                    # embedding and add a 1 mask value (following numpy.ma conventions)
                    X.append(np.zeros((emb_size,)))
                    mask.append(1)

                # Reset the embedding list and update the upper bound for this slot
                curr_slot_emb_list = []
                curr_upper_ts += delta

                # Add the current embedding to the new list
                curr_slot_emb_list += list(f['openl3'][idx]['openl3'])

        # Handle last time slot
        if curr_slot_emb_list:
            slot_emb = aggr_func(curr_slot_emb_list)
            X.append(slot_emb)
            mask.append(1)
        else:
            X.append(np.zeros((emb_size,)))
            mask.append(0)

    # Convert to np.array
    X = np.array(X)
    mask = np.array(mask)

    return X, mask


def series_splitter(mask, test_ratio=0.25, mode='random_holes', hole_mean=20, hole_std=10, min_hole_size=2):
    '''
    Given a `invalid_mask` of length n, generate masks for the train split and test split according to `test_ratio`.

    Params:
    -------
        mask: list or np.array
            an array of 1's and 0's. 1 being masked, 0 being readable
        test_ratio: float (0, 1)
            the fraction of the data to be split out as test
        mode: str
            should be one of 'random_holes', 'r', 'chronological', or 'c'.
            'random_holes' - punch_out randomly (Guassian according to `hole_param`) sized holes in the series and 
            set aside as test.
            'chronological' - just use the begining as train and the end as test, modeling real world delopyment.
            hole_param in chronological mode is ignored.
        hole_mean: positive number
            used for the 'random_holes' mode and ignored otherwise. Specify the mean of the normal distribution from 
            which the size of the hole is drawn.
        hole_std: positive number
            used for the 'random_holes' mode and ignored otherwise. Specify the mean of the normal distribution from 
            which the size of the hole is drawn.
        min_hole_size: positive number
            used for the 'random_holes' mode and ignored otherwise. size of the minimal hole.
    Returns:
    --------
        train_mask: np.array of int's
            an array of 1's and 0's that can be used to mask any series. 1 being masked, 0 being selected for the train split
        test_mask: np.array of int's
            an array of 1's and 0's that can be used to mask any series. 1 being masked, 0 being selected for the test split
    '''
    if mode == 'chronological' or mode == 'c':
        chrono_mode = True
    elif mode == 'random_holes' or mode == 'r':
        chrono_mode = False
    else:
        raise ValueError("type must be one of 'random_holes', 'r', 'chronological', or 'c'.")

    invalid_idx = np.array(mask, dtype=np.bool)
    valid_idx = 1 - invalid_idx
    valid_idx = valid_idx.astype(np.bool)
    n_total_valid = sum(valid_idx)
    n_test = int(round(test_ratio * n_total_valid))
    n_train = n_total_valid - n_test
    
    train_mask = np.zeros_like(mask)
    test_mask = np.zeros_like(mask)
    if chrono_mode:
        # first split the data as if all the valid frames are contiguous
        continuous_split_test = np.array([1] * n_train + [0] * n_test, dtype=int)
        continuous_split_train = 1 - continuous_split_test
        # then use the valid_mask to put these contiguous masks into the right places
        test_mask[valid_idx]  = continuous_split_test
        train_mask[valid_idx] = continuous_split_train

    else:
        # 'random_holes' mode
        ## first generate an array that holes the sizes of the holes, with length n_holes
        hole_sizes = []
        while sum(hole_sizes) < n_test:
            rand_size = int(round(np.random.normal(hole_mean, hole_std)))
            if rand_size >= min_hole_size:
                hole_sizes.append(rand_size)

        if sum(hole_sizes) > n_test:
            hole_sizes.pop()
            last_size = n_test - sum(hole_sizes)
            hole_sizes.append(last_size)

        ## next, decide where the holes are gonna be, ie, how to split the training data into n_holes + 1 parts
        n_holes = len(hole_sizes)
        # pick n_holes positions from all n_train+1 positions: this avoids two adjacent holes
        hole_start_positions = np.sort(np.random.choice(n_train+1, n_holes, replace=False))
        inter_hole_intervals = [hole_start_positions[0]] + list(np.diff(hole_start_positions))
        after_last_hole = n_train - hole_start_positions[-1]

        ## then build the continuous_split arrays according to hole_sizes
        continuous_split_test = []
        for hole, bread in zip(hole_sizes, inter_hole_intervals):
            continuous_split_test += bread * [1]
            continuous_split_test += hole * [0]
        continuous_split_test += after_last_hole * [1]

        ## finally put things into places
        continuous_split_test = np.array(continuous_split_test, dtype=int)
        continuous_split_train = 1 - continuous_split_test
        test_mask[valid_idx]  = continuous_split_test
        train_mask[valid_idx] = continuous_split_train

    return train_mask, test_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_path')
    parser.add_argument('output_path', default='.')
    parser.add_argument('delta_mins', type=int, default=15)
    # Aggregation mode flag
    parser.add_argument("-a", "--aggr", required=True,
                        choices=['random','centroid','medoid','anti_medoid'],
                        help="Aggregation method to use")

    args = parser.parse_args()

    if args.aggr == 'random':
        aggr_func = data_aggr.random
    if args.aggr == 'centroid':
        aggr_func = data_aggr.centroid
    if args.aggr == 'medoid':
        aggr_func = data_aggr.medoid
    if args.aggr == 'anti_medoid':
        aggr_func = data_aggr.anti_medoid


    out_fname = "{}_{}minslot_{}.npz".format(os.path.basename(args.hdf5_path).split('.')[0],
                                             args.delta_mins,
                                             args.aggr)
    out_path = os.path.join(args.output_path, out_fname)


    X, mask = load_openl3_time_series(args.hdf5_path, delta_mins=args.delta_mins, aggr_func=aggr_func)
    np.savez_compressed(out_path, X=X, mask=mask)
