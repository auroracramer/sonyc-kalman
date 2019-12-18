import os
import json
import itertools
from collections import OrderedDict
import hashlib

# Define constants.
script_name = "train_kvae.py"
script_path = os.path.abspath(os.path.join("..", script_name))
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Create folder.
sbatch_dir = os.path.join(script_name[:-3], "sbatch")
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = os.path.join(script_name[:-3], "slurm")
os.makedirs(slurm_dir, exist_ok=True)

sbatch_paths = []

data_dir = "/scratch/jtc440/sonyc-kalman"
data_path = os.path.join(data_dir, 'sonycnode-b827eb2a1bce_60minslot_medoid.npz')
train_mask_path = os.path.join(data_dir, 'sonycnode-b827eb2a1bce_60minslot_medoid_train_mask.npy')
test_mask_path = os.path.join(data_dir, 'sonycnode-b827eb2a1bce_60minslot_medoid_test_mask.npy')
output_dir = os.path.join(data_dir, "output")


defaults = {
    'n_timesteps': 48, ##
    'hop_length': 6, ##
    'init_lr': 0.007, ##
    'scale_reconstruction': 0.3, ##
    'kf_update_steps': 10, ##
    'num_epochs': 300,
    'train_miss_prob': 0.1, ##
    'decay_rate': 0.85,
    'decay_steps': 20,
    'out_distr': "gaussian",
    'K': 3, ##
    'batch_size': 64,
    't_steps_mask': 5,
    't_init_mask': 5,
    'dim_z': 16, ##
    'dim_a': 32, ##
    'alpha_units': 50, ##
    'noise_emission': 0.03, ##
    'noise_transition': 0.08, ##
    'init_cov': 20.0, ##
    'vae_num_units': 25, ##
    'num_layers': 2, ##
    'noise_var': 0.1, ##
    'll_keep_prob': 1.0,
    'generate_step': 5,
}


search_space = OrderedDict((
    ('n_timesteps', (24, 48, 72)),
    ('scale_reconstruction', (0.1, 0.3)),
    ('train_miss_prob', (0.0, 0.1)),
    ('K', (1, 4, 8)),
    ('dim_z', (16, 32, 64)),
    ('dim_a', (32, 64, 128)),
    ('alpha_units', (50, 100)),
    ('vae_num_units', (25, 50)),
    ('num_layers', (2, 4, 6)),
))

hash_to_params = {}

for params in itertools.product(*search_space.values()):

    script_args = [
        '--data_path', data_path,
        '--train_mask_path', train_mask_path,
        '--test_mask_path', test_mask_path
    ]
    args_dict = {}

    for param_name, param_value in zip(search_space.keys(), params):
        script_args += ["--" + param_name, str(param_value)]
        args_dict[param_name] = param_value

    for param_name, default_value in defaults.items():
        if param_name not in search_space:
            script_args += ["--" + param_name, str(default_value)]
            args_dict[param_name] = default_value

    # Sanity checks
    if args_dict['n_timesteps'] < args_dict['hop_length']:
        continue
    if args_dict['dim_a'] <= args_dict['dim_z']:
        continue

    trial_str = "_".join(['-'.join((x[0].replace('_', '-'), str(x[1])))
                          for x in sorted(args_dict.items(), key=lambda y: y[0])])

    hash_trial = str(hashlib.md5(trial_str.encode()).hexdigest())

    hash_to_params[hash_trial] = args_dict


with open('hashes_to_params.json', 'w') as f:
    json.dump(hash_to_params, f)
