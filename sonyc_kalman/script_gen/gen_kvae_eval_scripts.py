import os
import itertools
from collections import OrderedDict
import hashlib
import pandas as pd

# Define constants.
script_name = "eval_kvae.py"
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

gb_per_thread = 20
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
    f.write("#SBATCH --time=1:30:00\n")
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


for exp_hash in os.listdir(output_dir):
    exp_dir = os.path.join(output_dir, exp_hash)
    timestamp = sorted(os.listdir(exp_dir))[-1]
    exp_dir = os.path.join(exp_dir, timestamp)

    results_path = os.path.join(exp_dir, 'results.csv')
    imputation_results_path = os.path.join(exp_dir, 'imputation_results.json')

    if not os.path.exists(results_path):
        continue

    if os.path.exists(imputation_results_path):
        continue

    results = pd.read_csv(results_path)
    if len(results) == 0:
        continue

    final_results = dict(results.iloc[-1])
    if final_results['epoch'] < 299:
        continue

    model_path = os.path.join(exp_dir, 'model.ckpt')

    script_args = [
        '--data_path', data_path,
        '--train_mask_path', train_mask_path,
        '--test_mask_path', test_mask_path,
        '--reload_model', model_path,
    ]

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
