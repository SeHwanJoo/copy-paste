import os
from collections import defaultdict, Counter
import random

import numpy as np
import torch
import pandas as pd


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()

    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def seed_everything(seed=42):
    """Seed All

    Args:
        seed: seed number
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    """Get device type

    Returns: device, "cpu" if cuda is available else "cuda"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# get folds
def get_folds(df, config):
    df_folds = df[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds['fold'] = 0

    for fold, (trn_idx, val_idx) in enumerate(
            stratified_group_k_fold(df, df['class_id'], df['image_id'], config.n_folds, config.seed)):
        trn_ids = df.loc[trn_idx, 'image_id'].unique()
        val_ids = df.loc[val_idx, 'image_id'].unique()
        assert len(set(trn_ids).intersection(set(val_ids))) == 0

        df_folds.loc[val_ids, 'fold'] = fold

    return df_folds


def load_all_data(config):
    meta_df = pd.read_csv(os.path.join(config.root_path, config.origin_path, "train.csv"))
    meta_df = meta_df.merge(pd.read_csv(os.path.join(config.image_path, "train_meta.csv")),
                            on="image_id", how='left')

    meta_df = meta_df.reset_index(drop=True)
    meta_df = meta_df[meta_df['class_id'] != 14].reset_index(drop=True)

    if config.image_size == 'original':
        meta_df = meta_df[meta_df['class_id'] != 14].reset_index(drop=True)
        meta_df['x_min_resized'] = (meta_df['x_min']).astype(int)
        meta_df['x_max_resized'] = (meta_df['x_max']).astype(int)
        meta_df['y_min_resized'] = (meta_df['y_min']).astype(int)
        meta_df['y_max_resized'] = (meta_df['y_max']).astype(int)
    else:
        meta_df['x_min_resized'] = (meta_df['x_min'] * config.image_size / meta_df['dim1']).astype(int)
        meta_df['x_max_resized'] = (meta_df['x_max'] * config.image_size / meta_df['dim1']).astype(int)
        meta_df['y_min_resized'] = (meta_df['y_min'] * config.image_size / meta_df['dim0']).astype(int)
        meta_df['y_max_resized'] = (meta_df['y_max'] * config.image_size / meta_df['dim0']).astype(int)

    return meta_df




def load_normal_data(config):
    meta_df = pd.read_csv(os.path.join(config.root_path, config.origin_path, "train.csv"))
    meta_df = meta_df.merge(pd.read_csv(os.path.join(config.image_path, "train_meta.csv")),
                            on="image_id", how='left')

    meta_df = meta_df.reset_index(drop=True)

    meta_df = meta_df[meta_df['class_id'] == 14].reset_index(drop=True)
    # meta_df['x_min_resized'] = (meta_df['x_min'] * config.image_size / meta_df['dim1']).astype(int)
    # meta_df['x_max_resized'] = (meta_df['x_max'] * config.image_size / meta_df['dim1']).astype(int)
    # meta_df['y_min_resized'] = (meta_df['y_min'] * config.image_size / meta_df['dim0']).astype(int)
    # meta_df['y_max_resized'] = (meta_df['y_max'] * config.image_size / meta_df['dim0']).astype(int)

    return meta_df
