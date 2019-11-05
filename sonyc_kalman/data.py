import argparse
import datetime
import os
import pandas as pd
import h5py
import numpy as np

import data_aggr


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
        curr_upper_ts = datetime.datetime.utcfromtimestamp(timestamps[0]) + delta

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
                    mask.append(1)
                else:
                    # If there are no embeddings (i.e. there is missing data), add a dummy
                    # embedding and add a zero mask value
                    X.append(np.zeros((emb_size,)))
                    mask.append(0)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_path')
    parser.add_argument('output_path', default='.')
    parser.add_argument('delta_mins', type=int, default=15)
    # Aggregation mode flag
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-r', '--random', action='store_true')
    group.add_argument('-c', '--centroid', action='store_true')
    group.add_argument('-m', '--medoid', action='store_true')
    group.add_argument('-a', '--anti_medoid', action='store_true')

    args = parser.parse_args()
    
    aggr_func = None
    if args.random:
        aggr_func = data_aggr.random
    if args.centroid:
        aggr_func = data_aggr.centroid
    if args.medoid:
        aggr_func = data_aggr.medoid
    if args.anti_medoid:
        aggr_func = data_aggr.anti_medoid
        

    out_fname = "{}_{}minslot.npz".format(os.path.basename(args.hdf5_path).split('.')[0],
                                          args.delta_mins)
    out_path = os.path.join(args.output_path, out_fname)

    
    X, mask = load_openl3_time_series(args.hdf5_path, delta_mins=args.delta_mins, aggr_func=aggr_func)
    np.savez_compressed(out_path, X=X, mask=mask)
