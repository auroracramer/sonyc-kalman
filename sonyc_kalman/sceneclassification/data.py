import os
import numpy as np


def load_data(data_dir, ignore_list=None, return_metadata=False):
    """ Load TAU Urban Acoustic Scenes data """

    if ignore_list is None:
        ignore_list = set()
    else:
        ignore_list = set(ignore_list)

    # Initialize lists
    X = []
    scene_list = []
    city_list = []
    location_list = []
    segment_list = []
    device_list = []

    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)

        # Load embedding
        emb = np.load(path)

        # Obtain metadata from filename
        # Format:[scene label]-[city]-[location id]-[segment id]-[device id].npy
        scene, city, location, segment, device \
            = os.path.splitext(fname)[0].split('-')

        # Ignore specified scenes
        if scene in ignore_list:
            continue

        X.append(emb)
        scene_list.append(scene)
        city_list.append(city)
        location_list.append(location)
        segment_list.append(segment)
        device_list.append(device)

    scene_labels = sorted(list(set(scene_list)))

    # Convert data to numpy arrays suitable for models
    X = np.array(X)
    y = np.array([scene_labels.index(x) for x in scene_list])

    if not return_metadata:
        return X, y
    else:
        return (X, y, scene_labels, city_list,
                location_list, segment_list, device_list)
