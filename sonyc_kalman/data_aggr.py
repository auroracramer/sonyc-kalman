import numpy as np
from scipy.spatial import distance


def random(feature_list):
    ''' Sample a random feature vector for a list of feature vectors
    '''
    num_feat = len(feature_list)
    return np.asarray(feature_list[np.random.randint(num_feat)])

def centroid(feature_list):
    ''' Compute the centroid of a list of features
    '''
    return np.asarray(np.mean(feature_list, axis=0))

def medoid(feature_list, d_func=distance.euclidean):
    ''' Compute the medoid of a list of features using a specific distance function
    '''
    center = centroid(feature_list)
    d_to_center = [d_func(feature, center) for feature in feature_list]
    return np.asarray(feature_list[np.argmin(d_to_center)])

def anti_medoid(feature_list, d_func=distance.euclidean):
    ''' Compute the sample with the maximum distance from the centroid
    '''
    center = centroid(feature_list)
    d_to_center = [d_func(feature, center) for feature in feature_list]
    return np.asarray(feature_list[np.argmax(d_to_center)])
