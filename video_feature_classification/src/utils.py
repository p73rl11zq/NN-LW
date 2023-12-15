import sys

sys.path.append("../")

import os
import pickle
import random
import cv2 as cv

from pathlib import Path
from tqdm import tqdm

import numpy as np

from src.accuracy import Accuracy
from pytorch.common.datasets_parsers.av_parser import AVDBParser
from src.features import feature_extractor
from src.models import set_model

def get_data(dataset_root, file_list, max_num_clips=0): # max_num_samples=5000
    dataset_parser = AVDBParser(
        dataset_root,
        os.path.join(dataset_root, file_list),
        max_num_clips=max_num_clips,
        #max_num_samples=max_num_samples,
        ungroup=False,
        load_image=True,
    )
    data = dataset_parser.get_data()
    print("clips count:", len(data))
    print("frames count:", dataset_parser.get_dataset_size())
    return data

def calc_features(data):
    feat, targets = feature_extractor(data, mode = 1)
    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    if pca_dim > 0:
        pca = PCA(n_components = pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # mode is to be in ['rf', 'sgd', 'mlp', 'svc']
    y_pred = set_model(X_train, X_test, y_train, mode = 'sgd')

    print('')
    print('     ***  Metrics by frames    ***   ')
    accuracy_fn.by_frames(y_pred)
    print('')
    print('     ***  Metrics by clips    ***   ')
    accuracy_fn.by_clips(y_pred)
    return y_pred