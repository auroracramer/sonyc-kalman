import argparse
import sys
import os
import pickle

import pykalman as pk
from numpy import ma
from data import load_openl3_time_series as load_data


DEFAULT_SENSOR = '/beegfs/work/sonyc/features/openl3/2017/sonycnode-b827ebefb215.sonyc_features_openl3.h5'

def process_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--sensor',
                        dest='sensor', type=str, default=DEFAULT_SENSOR,
                        help='Location of sensor .h5 file containing l3 embeddings')
    
    parser.add_argument('--latent_dim',
                        dest='latent_dim', type=int, default=2,
                        help='Dimensionality of the latent space')
    
    parser.add_argument('--n_iter',
                        dest='n_iter', type=int, default=5,
                        help='Number of iterations in EM training')
    
    parser.add_argument('--data_range',
                        dest='data_range', type=int, default=None,
                        help='Upper index bound on data')
    
    parser.add_argument('--model_name', type=str, default = 'pykalman',
                        help='Name to store model under')
    
    parser.add_argument('--output_path', type=str,
                        help='Path to store trained pykalman output')
    
    return parser.parse_args(args)

if __name__=='__main__':
    params = process_arguments(sys.argv[1:])
    sensor_name = params.sensor.split('/')[-1]
    
    print('Loading data from {}'.format(sensor_name))
    data, mask = load_data(params.sensor)
    mask = (mask==0).astype('int') #due to masked data being denoted by 1,
                                   #opposite of numpy convention
    data[mask] = ma.masked
    if params.data_range is not None:
        data = data[:params.data_range] 
    
    print('Training Kalman Filter: Sensor: {},\t N_Iterations: {},\t Latent Space Dim{}'\
          .format(sensor_name, params.n_iter, params.latent_dim))
        
    kf = pk.KalmanFilter(n_dim_state=params.latent_dim,\
                         n_dim_obs=data.shape[1],\
                         em_vars='all')
    kf_trained = kf.em(data, n_iter=params.n_iter)
    
    print('Training complete, saving result to {}'.format(params.output_path))
    
    with open(os.path.join(params.output_path, params.model_name+'.pkl'), 'wb') as fd:
        pickle.dump(kf_trained, fd)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        