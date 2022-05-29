from scipy.stats import moment
import cv2
from skimage import img_as_float
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
import pickle
from os.path import exists
import numpy as np
from skimage import io
from extract_features import extract_features as extract_features1
import functools
import operator
from skimage.filters import median, sobel, unsharp_mask
from scipy.sparse import coo_matrix
from sklearn.utils import resample
import random

class AiDetector:
    def __init__(self):
        self.classifier_file_path = 'classifier.sav'
        self.classifier = None
        self.image = None
        self.expert_mask = None
        self.step = 1
        self.image_path = None
        self.manual_path = None
        self.mask_path = None
        self.result_img = None

    def load(self, image_path, manual_path, mask_path):
        self.image_path = image_path
        self.manual_path = manual_path
        self.mask_path = mask_path

    def run(self):
        if self.classifier is None:
            self.load_classifier()
        if self.classifier is None:
            return
        self.predict_image()

    def stats(self):
        manual = img_as_float(cv2.imread(self.manual_path, cv2.IMREAD_GRAYSCALE))
        mask = img_as_float(cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE))
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        error_img = []
        for i in range(self.result_img.shape[0]):
            row = []
            for j in range(self.result_img.shape[1]):
                manual_value = manual[i][j]
                mask_value = mask[i][j]
                img_value = self.result_img[i][j]
                if mask_value == 1:
                    if manual_value == 1:
                        if img_value == 1:
                            tp += 1
                            row.append((0, 255, 0))
                        else:
                            fn += 1
                            row.append((0, 0, 255))
                    else:
                        if img_value == 1:
                            fp += 1
                            row.append((255, 0, 0))
                        else:
                            tn += 1
                            row.append((0, 0, 0))
            error_img.append(row)
        # Trwałość
        ppv = (tp + tn) / (tn + fn + tp + fp)
        # Czułość
        tpr = tp / (tp + fn)
        # Swoistość
        spc = tn / (fp + tn)
        # Średnia
        avg = (spc + tpr) / 2
        return ppv, tpr, spc, avg, np.array(error_img)

    def flatten(self, arr):
        return functools.reduce(operator.iconcat, arr, [])

    def hu_moments(self, data):
        moments = cv2.moments(np.vectorize(float)(data))
        hu_moments = cv2.HuMoments(moments)
        return self.flatten(hu_moments)

    def extract_features(self, image, cache=False, image_path=None):
        print('extract_features...')
        if cache and exists(f'{image_path.split("/")[-1]}_features.sav'):
            return self.load_object(f'{image_path.split("/")[-1]}_features.sav')
        features_list = extract_features1(image, self.step)
        if cache:
            self.cache_object(features_list, f'{image_path.split("/")[-1]}_features.sav')
        return features_list

    def get_labels(self, image):
        label = []
        rows = range(image.shape[0] // self.step)
        for i in rows:
            cols = range(image.shape[1] // self.step)
            for j in cols:
                label.append(image[i * self.step, j * self.step])
        return label

    def predict_image(self):
        print(f'predict_image...')
        self.step = 1
        self.image = img_as_float(cv2.imread(self.image_path,)[:, :, 1])
        self.image = self.preprocess_image(self.image)
        features = self.extract_features(self.image, cache=False, image_path=self.image_path)
        data = self.classifier.predict(features)
        self.result_img = data.reshape(self.image.shape)
        self.masking()

    def masking(self):
        mask = img_as_float(cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE))
        for i in range(self.result_img.shape[0]):
            for j in range(self.result_img.shape[1]):
                self.result_img[i][j] = self.result_img[i][j] if mask[i][j] != 0 else 0.0

    def calc_contrast(self, val, lo, hi):
        if val < lo:
            return 0
        if val > hi:
            return 1
        return (val - lo) / (hi - lo)

    def contrast_image(self, img, low, high):
        res = [[self.calc_contrast(x, low, high) for x in y] for y in img]
        return np.array(res)

    def increase_contrast(self, img):
        nonzero_photo = img[np.nonzero(img)]
        percentiles = np.percentile(nonzero_photo, (2, 99))
        return self.contrast_image(img, percentiles[0], percentiles[1])

    def detect_edges(self, img):
        img_sobel = sobel(img)
        perc = np.percentile(img_sobel, (1, 99))
        return self.contrast_image(img_sobel, perc[0], perc[1])

    def preprocess_image(self, image):
        image = median(image)
        image = unsharp_mask(image, radius=10, amount=2)
        image = median(image)
        return image

    def downsample(self, x, y):
        x = np.array(x)
        y = np.array(y)

        minority_x = x[y == 1]
        minority_y = y[y == 1]
        majority_x = x[y == 0]
        majority_y = y[y == 0]

        randomlist = random.sample(range(0, len(majority_y)), len(majority_y) - len(minority_y))

        majority_x = np.delete(majority_x, randomlist, 0)
        majority_y = np.delete(majority_y, randomlist, 0)

        x = np.concatenate((majority_x, minority_x), axis=0)
        y = np.concatenate((majority_y, minority_y), axis=0)
        return x, y

    def create_classifier(self):
        print('create_classifier...')
        self.step = 5
        list_img = ['data/images/01_h.jpg', 'data/manual/01_h.tif']
        list_img1 = ['data/images/03_h.jpg', 'data/manual/03_h.tif']
        list_img2 = ['data/images/04_h.jpg', 'data/manual/04_h.tif']

        x = []
        y = []
        for i in [list_img, list_img1, list_img2]:
            print(str(i))

            image = img_as_float(cv2.imread(i[0])[:,:,1])
            image = self.preprocess_image(image)
            expert_mask = img_as_float(cv2.imread(i[1], cv2.IMREAD_GRAYSCALE))
            x += self.extract_features(image, cache=False, image_path=i[0])
            y += self.get_labels(expert_mask)

        x, y = self.downsample(x, y)

        kfold = KFold(n_splits=5, shuffle=True, random_state=1)

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for train_index, test_index in kfold.split(x):
            for i in train_index:
                x_train.append(x[i])
                y_train.append(y[i])
            for h in test_index:
                x_test.append(x[h])
                y_test.append(y[h])

        self.classifier = DecisionTreeClassifier(criterion="entropy")
        self.classifier = self.classifier.fit(x_train, y_train)

        print("score ", self.classifier.score(x_test, y_test))
        print('completed')

    def load_classifier(self):
        print('load_classifier...')
        self.classifier = self.load_object(self.classifier_file_path)
        print('completed')

    def save_classifier(self):
        print('save_classifier...')
        self.cache_object(self.classifier, self.classifier_file_path)
        print('completed')

    def cache_object(self, object, name):
        pickle.dump(object, open(name, 'wb'))

    def load_object(self, name):
        if exists(name):
            file = open(name, 'rb')
            data = pickle.load(file)
            file.close()
            return data
        else:
            return None
