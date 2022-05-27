from scipy.stats import moment
import cv2
from skimage import img_as_float
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
import pickle
import numpy as np
from skimage import io

class AiDetector:
    def __init__(self):
        self.classifier_file_path = 'classifier.sav'
        self.classifier = None
        self.image = None
        self.expert_mask = None
        self.step = 1

    def get_features(self, image):
        features_list = []

        rows = range(image.shape[0] // self.step)
        for i in rows:
            cols = range(image.shape[1] // self.step)
            for j in cols:
                # part_img = image[5 * i:5 * (i + 1), 5 * j:5 * (j + 1)].flatten()
                # part_img = image[i * step: i * step + 5, j * step: j * step + 5].flatten()
                part_img = image[max(0, i * self.step - 2): i * self.step + 3, max(0, j * self.step - 2): j * self.step + 3].flatten()
                features = []
                features.append(moment(part_img, moment=2))
                features.append(moment(part_img, moment=3))
                features.append(moment(part_img, moment=4))
                features.append(moment(part_img, moment=5))
                features_list.append(features)
        return features_list

    def get_labels(self, image, step=5):
        label = []

        rows = range(image.shape[0] // step)
        for i in rows:
            cols = range(image.shape[1] // step)
            for j in cols:
                img_part = image[5 * i: 5 * (i + 1), 5 * j: 5 * (j + 1)]
                label.append(img_part[2, 2])
        return label

    def prepare(self, image, predicted, expert_mask):
        data = self.get_labels(expert_mask)
        x, y1 = image.shape
        pre = []
        hand = []
        k = 0
        for i in range(x // 5):
            pre.append([])
            hand.append([])
            for j in range(y1 // 5):
                pre[i] += [predicted[k]]
                hand[i] += [data[k]]
                k += 1
        return pre, hand

    def data(self, imgs):
        x = []
        y = []
        for i in imgs:
            print(str(i))

            image = img_as_float(cv2.imread(i[0], cv2.IMREAD_GRAYSCALE))
            expert_mask = img_as_float(cv2.imread(i[1], cv2.IMREAD_GRAYSCALE))
            x += self.get_features(image, self.step)
            y += self.get_labels(expert_mask, self.step)

        kfold = KFold(n_splits=5, shuffle=True, random_state=1)

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for train_index, test_index in kfold.split(x):
            for i in train_index:
                X_train.append(x[i])
                y_train.append(y[i])
            for h in test_index:
                X_test.append(x[h])
                y_test.append(y[h])

        clf = DecisionTreeClassifier(criterion="entropy")
        clf = clf.fit(X_train, y_train)
        predictions = cross_val_predict(clf, x, y, cv=6)
        print("predictions:", predictions)
        print("score ", clf.score(X_test, y_test))

        return clf

    def predict(self, img_name):
        imgs = [f'data/images/{img_name}.jpg', f'data/manual/{img_name}.tif']

        self.image = img_as_float(cv2.imread(imgs[0], cv2.IMREAD_GRAYSCALE))
        self.expert_mask = img_as_float(cv2.imread(imgs[1], cv2.IMREAD_GRAYSCALE))
        print('\textract features...')
        x = self.get_features(self.image, self.step)
        # self.cache_object(x, 'features.sav')
        # x = self.load_object('features.sav')
        print('\tcomplete')
        # y = self.get_labels(expert_mask)
        # print('score', self.classifier.score(x, y))
        return self.classifier.predict(x)

    def predict_image(self, img_name):
        print(f'predict_image {img_name}...')
        data = self.predict(img_name)
        predicted_image = np.zeros(self.image.shape)
        rows = range(predicted_image.shape[0] // self.step)
        for i in rows:
            cols = range(predicted_image.shape[1] // self.step)
            for j in cols:
                predicted_image[i * self.step, j * self.step] = data[i * len(cols) + j]

        io.imsave('predicted_image.jpg', predicted_image)

    def create_classifier(self):
        print('create_classifier...')
        one = ['data/images/01_h.jpg', 'data/manual/01_h.tif']
        two = ['data/images/02_h.jpg', 'data/manual/02_h.tif']
        three = ['data/images/03_h.jpg', 'data/manual/03_h.tif']
        four = ['data/images/04_h.jpg', 'data/manual/04_h.tif']
        five = ['data/images/05_h.jpg', 'data/manual/05_h.tif']

        # clf = self.data([one, two, three, four, five])
        self.classifier = self.data([one])
        print('completed')

        # img_data = two
        # full_img_df = prepare_data(img_data, 1)
        # labels = clf.predict(full_img_df.loc[:, full_img_df.columns != 'label'])
        #
        # full_img_labels = normalize(cv2.imread(img_data[1], cv2.IMREAD_GRAYSCALE))
        # io.imshow(cv2.imread(img_data[0], cv2.IMREAD_GRAYSCALE))
        # io.show()
        # io.imshow(reshape_labels(labels, full_img_labels.shape))

    def load_classifier(self):
        print('load_classifier...')
        self.classifier = pickle.load(open(self.classifier_file_path, 'rb'))
        print('completed')

    def save_classifier(self):
        print('save_classifier...')
        pickle.dump(self.classifier, open(self.classifier_file_path, 'wb'))
        print('completed')

    def cache_object(self, object, name):
        pickle.dump(object, open(name, 'wb'))

    def load_object(self, name):
        return pickle.load(open(name, 'rb'))