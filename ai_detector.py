from scipy.stats import moment
import cv2
from skimage import img_as_float
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
import pickle
from os.path import exists
import numpy as np
from skimage import io

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

    def extract_features(self, image, cache=False):
        if cache and exists(f'{self.image_path.split("/")[-1]}_features.sav'):
            return self.load_object(f'{self.image_path.split("/")[-1]}_features.sav')
        print('extract_features...')
        features_list = []

        rows = range(image.shape[0] // self.step)
        for i in rows:
            print(f'extract_features {i}/{len(rows)} {i/len(rows)*100}%')
            cols = range(image.shape[1] // self.step)
            for j in cols:
                part_img = image[max(0, i * self.step - 2): i * self.step + 3, max(0, j * self.step - 2): j * self.step + 3].flatten()
                features = []
                features.append(moment(part_img, moment=2))
                features.append(moment(part_img, moment=3))
                features.append(moment(part_img, moment=4))
                features.append(moment(part_img, moment=5))
                features_list.append(features)
        if cache:
            self.cache_object(features_list, f'{self.image_path.split("/")[-1]}_features.sav')
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
        self.image = img_as_float(cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE))
        features = self.extract_features(self.image, cache=True)
        data = self.classifier.predict(features)
        self.result_img = data.reshape(self.image.shape)
        self.masking()

    def masking(self):
        mask = img_as_float(cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE))
        for i in range(self.result_img.shape[0]):
            for j in range(self.result_img.shape[1]):
                self.result_img[i][j] = self.result_img[i][j] if mask[i][j] != 0 else 0.0

    def create_classifier(self):
        print('create_classifier...')
        self.step = 5
        one = ['data/images/01_h.jpg', 'data/manual/01_h.tif']
        two = ['data/images/02_h.jpg', 'data/manual/02_h.tif']
        three = ['data/images/03_h.jpg', 'data/manual/03_h.tif']
        four = ['data/images/04_h.jpg', 'data/manual/04_h.tif']
        five = ['data/images/05_h.jpg', 'data/manual/05_h.tif']
        test1 = ['data/images/02_h_800.jpg', 'data/manual/02_h_800.tif']
        test2 = ['data/images/02_h_800.jpg', 'data/manual/02_h_800.tif']

        x = []
        y = []
        for i in [one]:
            print(str(i))

            image = img_as_float(cv2.imread(i[0], cv2.IMREAD_GRAYSCALE))
            expert_mask = img_as_float(cv2.imread(i[1], cv2.IMREAD_GRAYSCALE))
            x += self.extract_features(image)
            y += self.get_labels(expert_mask)

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
        predictions = cross_val_predict(self.classifier, x, y, cv=6)

        print("predictions:", predictions)
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
            return pickle.load(open(name, 'rb'))
        else:
            return None
