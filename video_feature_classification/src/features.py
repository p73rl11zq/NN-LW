import scipy as sc
from time import time
import os
import pickle
import random
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
import numpy as np

def feature_extractor(data, mode = 1):

    feat, targets = [], []
    progresser = tqdm(iterable=range(0, len(data)),
                      desc='calc video features',
                      total=len(data),
                      unit='files')

    if mode == 1:
        radius = 5
        padding = radius * 2
        orb = cv.ORB_create(edgeThreshold=0)

        for i in progresser:
            clip = data[i]
            for sample in clip.data_samples:
                image = np.pad(sample.image, ((padding * 2, padding * 2), (padding * 2, padding * 2), (0, 0)),
                            'constant', constant_values=0)
                image = image.astype(dtype=np.uint8)
                kps = []
                for landmark in sample.landmarks:
                    kps.append(cv.KeyPoint(
                        landmark[0] + padding, landmark[1] + padding, radius))
                kp, descr = orb.compute(image, kps)

                pwdist = sc.spatial.distance.pdist(np.asarray(sample.landmarks))
                feat.append(np.hstack((descr.ravel(), pwdist.ravel())))

                targets.append(clip.labels)
    
    elif mode == 2:
        orb = cv2.ORB_create()

        for i in progresser:
            clip = data[i]
            rm_list = []
            for sample in clip.data_samples:
                '''
                print(f'sample.labels: {sample.labels}')
                if sample.labels != 8 or sample.labels != 7:
                    continue
                    print('yeeee')
                else:
                    pass
                '''
                # make image border
                bordersize = 15
                border = cv2.copyMakeBorder(sample.image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                            borderType=cv2.BORDER_CONSTANT, value=[0]*3)
                # make keypoint list
                keypoints = []
                for k in range(18, 68):
                    keypoints.append(cv2.KeyPoint(x=sample.landmarks[k][0]+bordersize,
                                                    y=sample.landmarks[k][1]+bordersize,
                                                    size=128))
                # compute the descriptors with ORB
                keypoints_actual, descriptors = orb.compute(border, keypoints)
                if len(keypoints_actual) != len(keypoints):
                    rm_list.append(sample)
                    continue

                descriptors = np.concatenate(descriptors)
                feat.append(descriptors)
                targets.append(sample.labels)

            for sample in rm_list:
                clip.data_samples.remove(sample)

    return feat, targets