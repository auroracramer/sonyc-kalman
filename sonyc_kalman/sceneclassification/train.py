import argparse
import os
import datetime
import json
import numpy as np
import random
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .data import load_data


def train_model(data_dir, output_dir, valid_ratio=0.1, weight_decay=1.0,
                n_cpus=1, ignore_classes=None, random_seed=0):
    # Set random seed for reproducability
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Create a new directory for this experiment
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(output_dir, ts)
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment params
    params_path = os.path.join(output_dir, 'params.json')
    params = {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'weight_decay': weight_decay,
        'n_cpus': n_cpus,
        'ignore_classes': ignore_classes
    }
    with open(params_path, 'w') as f:
        json.dump(params, f)

    # Load data and construct model
    X, y = load_data(data_dir, ignore_list=ignore_classes)
    m = LogisticRegression(C=weight_decay, n_jobs=n_cpus, solver='lbfgs')

    # Create train/valid split
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio,
                                         random_state=random_seed)
    train_idxs, valid_idxs = next(iter(strat_split.split(X, y)))
    X_train, y_train = X[train_idxs], y[train_idxs]
    X_valid, y_valid = X[valid_idxs], y[valid_idxs]

    # Train and save model
    m.fit(X_train, y_train)
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(m, model_path)

    # Get accuracy scores
    y_train_pred = m.predict(X_train)
    y_valid_pred = m.predict(X_valid)

    # Compute metrics and save
    metrics = {
        'train_micro_acc': accuracy_score(y_train, y_train_pred),
        'valid_micro_acc': accuracy_score(y_valid, y_valid_pred),
        'train_macro_acc': balanced_accuracy_score(y_train, y_train_pred),
        'valid_macro_acc': balanced_accuracy_score(y_valid, y_valid_pred)
    }
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1.0)
    parser.add_argument('--n-cpus', type=int, default=1)
    parser.add_argument('--ignore-classes', type=str, nargs="+")
    parser.add_argument('--random-seed', type=int, default=0)
    args = parser.parse_args()

    train_model(args.data_dir,
                output_dir=args.output_dir,
                valid_ratio=args.valid_ratio,
                weight_decay=args.weight_decay,
                n_cpus=args.n_cpus,
                ignore_classes=args.ignore_classes,
                random_seed=args.random_seed)
