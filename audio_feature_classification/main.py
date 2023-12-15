import sys

sys.path.append("../")

import pickle
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.utils import get_data, calc_features, classification
from src.accuracy import Accuracy
from pytorch.common.datasets_parsers.av_parser import AVDBParser


if __name__ == "__main__":
    experiment_name = "exp_1"
    max_num_clips = 0  # загружайте только часть данных для отладки кода (0 - все данные)
    use_dump = False  # используйте dump для быстрой загрузки рассчитанных фич из файла

    # dataset dir
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

    # opensmile configuration
    opensmile_root_dir = Path("/home/galahad/Documents/6_course/neural_networks/opensmile")
    # TODO: поэкспериментируйте с различными конфигурационными файлами библиотеки OpenSmile
    # opensmile_config_path = opensmile_root_dir / "config/avec11-14/avec2013.conf"
    opensmile_config_path = opensmile_root_dir / "config/egemaps/v02/eGeMAPSv02.conf"

    if not use_dump:
        # load dataset
        train_data = get_data(
            train_dataset_root, train_file_list, max_num_clips=max_num_clips
        )
        test_data = get_data(
            test_dataset_root, test_file_list, max_num_clips=max_num_clips
        )

        # get features
        train_feat, train_targets = calc_features(
            train_data, opensmile_root_dir, opensmile_config_path
        )
        test_feat, test_targets = calc_features(
            test_data, opensmile_root_dir, opensmile_config_path
        )

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        with open(experiment_name + ".pickle", "wb") as f:
            pickle.dump(
                [train_feat, train_targets, test_feat, test_targets, accuracy_fn],
                f,
                protocol=2,
            )
    else:
        with open(experiment_name + ".pickle", "rb") as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(
                f
            )

    # run classifiers
    classification(
        train_feat,
        test_feat,
        train_targets,
        test_targets,
        accuracy_fn=accuracy_fn,
        pca_dim=0,
    )
