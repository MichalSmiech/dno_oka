from scipy.stats import moment
import cv2
from skimage import img_as_float
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
import pickle
from os.path import exists
import numpy as np
from skimage import io
from multiprocessing import Pool
import tqdm
import time
import random
from scipy.stats import moment
import cv2
from skimage import img_as_float
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
import pickle
from os.path import exists
import numpy as np
from skimage import io
import functools
import operator
from skimage.filters import median, sobel, unsharp_mask
from scipy.sparse import coo_matrix
from sklearn.utils import resample
import random


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


if __name__ == '__main__':
    image_path = 'data/images/05_h.jpg'
    image = img_as_float(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    rows = list(range(image.shape[0] // step))
    random.shuffle(rows)

    data = list(zip([image] * len(rows), rows))
    with Pool(8) as p:
      features = list(tqdm.tqdm(p.imap(extract_features_row, data), total=len(data)))
    cache_object(features, f'{image_path.split("/")[-1]}_features.sav')

    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 10000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)