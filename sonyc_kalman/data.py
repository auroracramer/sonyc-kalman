import argparse
import datetime
import itertools
import operator
import os
import pandas as pd
import h5py
import numpy as np
import librosa

try:
    import data_aggr
except:
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


def series_splitter(mask_length, test_ratio=0.25, mode='random_holes', hole_mean=20, hole_std=10, min_hole_size=2, random_state=0):
    '''
    Given a `invalid_mask` of length n, generate masks for the train split and test split according to `test_ratio`.

    Params:
    -------
        mask_length: int
            length of the mask
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
        random_state: int
            for reproducability

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

    np.random.seed(random_state)

    n_test = int(round(test_ratio * mask_length))
    n_train = mask_length - n_test

    # build test_mask as list
    test_mask = list()
    if chrono_mode:
        test_mask = [1] * n_train + [0] * n_test
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
        ## pick n_holes positions from all n_train+1 positions: this avoids two adjacent holes
        hole_start_positions = np.sort(np.random.choice(n_train+1, n_holes, replace=False))
        inter_hole_intervals = [hole_start_positions[0]] + list(np.diff(hole_start_positions))
        after_last_hole = n_train - hole_start_positions[-1]

        ## then build the continuous_split arrays according to hole_sizes
        for hole, bread in zip(hole_sizes, inter_hole_intervals):
            test_mask += bread * [1]
            test_mask += hole * [0]
        test_mask += after_last_hole * [1]

    # convert and clean up
    test_mask = np.array(test_mask, dtype=int)
    train_mask = 1 - test_mask

    return train_mask, test_mask


def mask_to_segment_idxs(mask):
    '''
    Given a `mask`, return a list of arrays of indices for contiguous unmasked segments.

    Params:
    -------
        mask: np.array
            array of mask values

    Returns:
    --------
        segment_idxs_list: list of np.array of int's
            a list of arrays of indices for contigulous unmasked segments
    '''
    segment_idxs_list = []
    for key, it in itertools.groupby(enumerate(mask), key=operator.itemgetter(1)):
        # Skip masked values
        if key:
            continue

        segment_idxs_list.append(np.array(list(zip(*it))[0]))

    return segment_idxs_list


def construct_kvae_data(X, invalid_mask, subset_mask, n_timesteps=24, hop_length=6, random_state=0, test=False):
    '''
    Constructs KVAE friendly input matrix out of data matrix `X` and mask array `invalid_mask`.

    Each contiguous segment in `X` (as specified by `mask`) is divided into examples
    of size `n_timesteps` with a hop length of `hop_length`. Padding is performed
    to ensure that all frames are accounted for.

    Params:
    -------
        X: np.array of shape (num_frames, feature_dim)
            data matrix
        invalid_mask: np.array
            array of invalid mask values
        subset_mask: np.array
            array of mask values for the given subset
        n_timesteps: int
            number of frames per training example
        hop_length: int
            hop size for dividing each sequence into examples
        random_state: int
            for reproducability
        test: bool
            If true, do not split segments into frames


    Returns:
    --------
        X_frames: np.array of shape (num_examples, `n_timesteps`, feature_dim)
            array of training data suitable for input in KVAE model
        mask_frames: np.array of shape (num_examples, `n_timesteps`)
            array of mask values suitable for input in KVAE model
    '''
    X_list = []
    mask_list = []

    seg_idxs_list = mask_to_segment_idxs(subset_mask)
    max_seg_len = max([len(x) for x in seg_idxs_list])

    for seg_idxs in seg_idxs_list:
        # Extract the current segment
        X_seg = X[seg_idxs, :]
        mask_seg = invalid_mask[seg_idxs]

        num_frames = len(seg_idxs)
        if not test:
            pad_length = max(0, int(np.ceil((num_frames - n_timesteps)/hop_length))*hop_length) + n_timesteps - num_frames
        else:
            pad_length = max_seg_len - num_frames

        if pad_length > 0:
            # Pad the segment so we don't lose any frames
            X_seg = np.pad(X_seg, ((0, pad_length), (0,0)), mode='constant')
            # Padding is with ones for the mask
            mask_seg = np.pad(mask_seg, (0, pad_length), mode='constant', constant_values=1)

        if not test:
            # Divide segment into frames
            X_seg_frames = librosa.util.frame(X_seg, frame_length=n_timesteps, hop_length=hop_length, axis=0)
            mask_seg_frames = librosa.util.frame(mask_seg, frame_length=n_timesteps, hop_length=hop_length).T
        else:
            X_seg_frames = X_seg[np.newaxis, ...]
            mask_seg_frames = mask_seg[np.newaxis, ...]

        # Accumulate current segment batches
        X_list.append(X_seg_frames)
        mask_list.append(mask_seg_frames)

    X_list = np.concatenate(X_list, axis=0)
    mask_list = np.concatenate(mask_list, axis=0)
    shuffle_idxs = np.random.permutation(X_list.shape[0])

    return X_list[shuffle_idxs], mask_list[shuffle_idxs]


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
