import sys

sys.path.append("../")

import os
import pickle
import random

from pathlib import Path
from tqdm import tqdm

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from accuracy import Accuracy
from pytorch.common.datasets_parsers.av_parser import AVDBParser

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

experiment_name = "exp_1"
max_num_clips = 0  # загружайте только часть данных для отладки кода
use_dump = False  # используйте dump для быстрой загрузки рассчитанных фич из файла

mode = 1
base_dir = Path("/home/specialo0/notebooks/data/uchebnoe/neural_n/")

if mode == 1:
    train_dataset_root  = base_dir / "Ryerson/Video"
    train_file_list     = base_dir / "Ryerson/train_data_with_landmarks.txt"
    test_dataset_root   = base_dir / "Ryerson/Video"
    test_file_list      = base_dir / "Ryerson/test_data_with_landmarks.txt"
elif mode == 2:
    train_dataset_root  = (base_dir / "OMGEmotionChallenge/omg_TrainVideos/preproc/frames")
    train_file_list     = (base_dir / "OMGEmotionChallenge/omg_TrainVideos/train_data_with_landmarks.txt")
    test_dataset_root   = (base_dir / "OMGEmotionChallenge/omg_ValidVideos/frames")
    test_file_list      = (base_dir / "OMGEmotionChallenge/omg_ValidVideos/valid_data_with_landmarks.txt")
else:
    raise (ValueError('mode is to be in 1 or 2'))

train_data = get_data(train_dataset_root, train_file_list, max_num_clips=0)
test_data = get_data(test_dataset_root, test_file_list, max_num_clips=0)

def calc_features(data):
    progresser = tqdm(
        iterable=range(0, len(data)),
        desc="calc video features",
        total=len(data),
        unit="files",
    )

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        rm_list = []
        for sample in clip.data_samples:
            pass

            # TODO: придумайте способы вычисления признаков по изображению с использованием ключевых точек
            # используйте библиотеку OpenCV
            '''
            if sample.labels in [7, 8]:
                continue
            if i % 8 != 0:
                continue
            '''
            dist = []
            lm_ref = sample.landmarks[30]  # point on the nose
            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))
            feat.append(dist)
            targets.append(sample.labels)

        for sample in rm_list:
            clip.data_samples.remove(sample)

    print("feat count:", len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

# get features
train_feat, train_targets = calc_features(train_data)
test_feat, test_targets = calc_features(test_data)

accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    if pca_dim > 0:
        # TODO: выполните сокращение размерности признаков с использованием PCA\
        pca = PCA(n_components = pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # TODO: используйте классификаторы из sklearn
    rf = RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        max_depth = 6,
        min_samples_split = 5,
        min_samples_leaf = 3,
        random_state = 4
        )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy_fn.by_frames(y_pred)
    accuracy_fn.by_clips(y_pred)
    return y_pred

classification(
            train_feat,
            test_feat,
            train_targets,
            test_targets,
            accuracy_fn=accuracy_fn,
            pca_dim=0,
            )