import os
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


gb_per_thread = 16
cpus_per_task = 4
jobs_per_file = 16
threads_per_file = 4
mem_gb = min(gb_per_thread * threads_per_file, 120)


assert (jobs_per_file % threads_per_file == 0)


def start_sbatch_file(script_name, sbatch_dir, sbatch_idx):
    job_name = "{}_{}".format(script_name[:-3], sbatch_idx)
    file_name = job_name + ".sbatch"
    sbatch_path = os.path.join(sbatch_dir, file_name)

    f = open(sbatch_path, "w")
    f.write("#!/bin/bash\n")
    f.write("\n")
    f.write("#SBATCH --job-name=" + script_name[:-3] + "\n")
    f.write("#SBATCH --nodes=1\n")
#    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --tasks-per-node=1\n")

    f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
    f.write("#SBATCH --time=30:00:00\n")
    f.write("#SBATCH --mem={}GB\n".format(mem_gb))
#    f.write("#SBATCH --output=\"" + \
#            os.path.join(os.path.abspath(slurm_dir),
#                         job_name + "_%j.out") + "\"\n")
    f.write("#SBATCH --output=foo.out\n")
    f.write("#SBATCH --err=\"" + \
            os.path.join(os.path.abspath(slurm_dir),
                         job_name + "_%j.err") + "\"\n")
    f.write("\n")
#    if threads_per_file > 1:
#        f.write("#PRINCE PRINCE_GPU_MPS=YES")
#        f.write("\n")
    f.write("module purge\n")
#    f.write("module load cuda/8.0.44\n")
#    f.write("module load cudnn/8.0v6.0\n")
    f.write("source ~/.bashrc\n")
    f.write("source activate ust_gpu\n")
    f.write("\n")
    f.write("cd {}\n".format(project_dir))

    return sbatch_path, f


def close_sbatch_file(f):
    f.write("\n")
    f.write("wait\n")
    f.close()


job_idx = 0
sbatch_idx = 1
sbatch_path, f = start_sbatch_file(script_name, sbatch_dir, sbatch_idx)
sbatch_paths.append(sbatch_path)


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

    if job_idx == jobs_per_file:
        # Close previous sbatch file
        close_sbatch_file(f)

        # Update indices
        job_idx = 0
        sbatch_idx += 1

        # Start new file
        sbatch_path, f = start_sbatch_file(script_name, sbatch_dir,
                                           sbatch_idx)
        sbatch_paths.append(sbatch_path)

    trial_str = "_".join(['-'.join((x[0].replace('_', '-'), str(x[1])))
                          for x in sorted(args_dict.items(), key=lambda y: y[0])])

    hash_trial = str(hashlib.md5(trial_str.encode()).hexdigest())
    script_args += ["--log_dir", os.path.join(output_dir, hash_trial)]

    script_path_with_args = " ".join(
        [script_path] + script_args)

    if threads_per_file == 1:
        f.write("OMP_NUM_THREADS=1 python " + script_path_with_args + "\n")
    else:
        if (job_idx % threads_per_file) == 0:
            f.write('wait\n')
        f.write("OMP_NUM_THREADS=1 python " + script_path_with_args + " &\n")

    job_idx += 1


if not f.closed:
    close_sbatch_file(f)

filename = script_name[:-3] + "_launch"
filepath = os.path.join(sbatch_dir, filename + ".sh")
# Open shell file.
with open(filepath, "w") as f:
    f.write("#!/usr/bin/env bash\n")
    for sbatch_path in sbatch_paths:
        # Write SBATCH command to shell file.
        f.write("sbatch {}\n".format(os.path.abspath(sbatch_path)))

    # Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(filepath).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(filepath, mode)
