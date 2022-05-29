from multiprocessing import Pool
import tqdm
import cv2
import pickle
import numpy as np
import functools
import operator


def cache_object(object, name):
    pickle.dump(object, open(name, 'wb'))

def flatten(arr):
    return functools.reduce(operator.iconcat, arr, [])

def hu_moments(data):
    moments = cv2.moments(np.vectorize(float)(data))
    hu_moments = cv2.HuMoments(moments)
    return flatten(hu_moments)

def extract_features_row(data):
    image, i, step = data

    row_features = []

    cols = range(image.shape[1] // step)
    for j in cols:
        part_img = image[max(0, i * step - 2): i * step + 3,
                   max(0, j * step - 2): j * step + 3].flatten()
        features = []
        features.extend(hu_moments(part_img))
        features.extend(part_img)
        if len(part_img) < 25:
            features.extend([0] * (25 - len(part_img)))
        row_features.append(features)
    return row_features


def extract_features(image, step):
    rows = list(range(image.shape[0] // step))
    # random.shuffle(rows)

    data = list(zip([image] * len(rows), rows, [step] * len(rows)))
    with Pool(8) as p:
        features = list(tqdm.tqdm(p.imap(extract_features_row, data), total=len(data)))
    result = []
    for feature in features:
        result.extend(feature)
    return result