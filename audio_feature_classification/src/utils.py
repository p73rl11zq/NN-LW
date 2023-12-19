import sys

sys.path.append("../")

import pickle
import random

from pathlib import Path

import numpy as np


from tqdm import tqdm

from pytorch.common.datasets_parsers.av_parser import AVDBParser
from src.voice_feature_extraction import OpenSMILE
from src.accuracy import Accuracy
from src.models import *


def get_data(dataset_root, file_list, max_num_clips: int = 0):
    dataset_parser = AVDBParser(dataset_root, file_list, max_num_clips=max_num_clips)
    data = dataset_parser.get_data()
    print("clips count:", len(data))
    print("frames count:", dataset_parser.get_dataset_size())
    return data


def calc_features(data, opensmile_root_dir, opensmile_config_path):
    vfe = OpenSMILE(opensmile_root_dir, opensmile_config_path)

    progresser = tqdm(
        iterable=range(0, len(data)),
        desc="calc audio features",
        total=len(data),
        unit="files",
    )

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        try:
            voice_feat = vfe.process(clip.wav_rel_path)
            feat.append(voice_feat)
            targets.append(clip.labels)
        except Exception as e:
            print(f"error calc voice features! {e}")
            data.remove(clip)

    print("feat count:", len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    if pca_dim > 0:
        pca = PCA(n_components=pca_dim, svd_solver='full')
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        # pass
        # TODO: выполните сокращение размерности признаков с использованием PCA

    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    y_pred = set_model(X_train, X_test, y_train, mode = 'sgd')

    accuracy_fn.by_clips(y_pred)