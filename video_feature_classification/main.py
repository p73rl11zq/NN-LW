import sys

sys.path.append("../")

import os
import pickle
import random
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.utils import get_data, calc_features, classification
from src.accuracy import Accuracy


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
    train_dataset_root  = (base_dir / "OMGEmotionChallenge/omg_TrainVideos/frames")
    train_file_list     = (base_dir / "OMGEmotionChallenge/omg_TrainVideos/train_data_with_landmarks.txt")
    test_dataset_root   = (base_dir / "OMGEmotionChallenge/omg_ValidVideos/frames")
    test_file_list      = (base_dir / "OMGEmotionChallenge/omg_ValidVideos/valid_data_with_landmarks.txt")
else:
    raise (ValueError('mode is to be in 1 or 2'))

train_data = get_data(train_dataset_root, train_file_list, max_num_clips=0)
test_data = get_data(test_dataset_root, test_file_list, max_num_clips=0)

# get features
train_feat, train_targets = calc_features(train_data)
test_feat, test_targets = calc_features(test_data)

accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

classification(
            train_feat,
            test_feat,
            train_targets,
            test_targets,
            accuracy_fn=accuracy_fn,
            pca_dim=0,
            )