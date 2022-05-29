from skimage import io, draw
from skimage.filters import frangi, hessian, median, try_all_threshold, threshold_otsu, threshold_mean
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte
from skimage.morphology import opening, square, closing
import matplotlib.pyplot as plt
from skimage.data import camera
import numpy as np
from scipy import signal, misc

class SimpleDetector:
    def __init__(self):
        self.original_img = None
        self.manual = None
        self.mask = None
        self.work_img = None
        self.result_img = None

    def load(self, image_path, manual_path, mask_path):
        self.original_img = io.imread(image_path)
        self.work_img = self.original_img.copy()
        self.manual = io.imread(manual_path, as_gray=True)
        if self.manual.max() == 255:
            self.manual = np.array(self.manual) / 255
        self.mask = io.imread(mask_path, as_gray=True)

    def stats(self):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        error_img = []
        for i in range(self.result_img.shape[0]):
            row = []
            for j in range(self.result_img.shape[1]):
                manual = self.manual[i][j]
                mask = self.mask[i][j]
                img = self.work_img[i][j]
                if mask == 1:
                    if manual == 1:
                        if img == 1:
                            tp += 1
                            row.append((0, 255, 0))
                        else:
                            fn += 1
                            row.append((0, 0, 255))
                    else:
                        if img == 1:
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

    def pre_processing(self):
        self.work_img = self.work_img[:,:,1]
        self.work_img = equalize_adapthist(self.work_img)
        self.work_img = img_as_ubyte(self.work_img)
        self.work_img = median(self.work_img)
        # io.imsave('test.jpg', self.work_img)

    def segmentation(self):
        self.work_img = frangi(self.work_img)
        # self.work_img = closing(self.work_img, square(3))
        # self.work_img = opening(self.work_img, square(5))
        self.masking()
        self.work_img = median(self.work_img)
        # self.work_img = self.work_img * 100
        # self.convolution2d()
        # self.work_img = img_as_ubyte(self.work_img)
        # io.imsave('test.jpg', self.work_img)

    def masking(self):
        for i in range(self.work_img.shape[0]):
            for j in range(self.work_img.shape[1]):
                self.work_img[i][j] = self.work_img[i][j] if self.mask[i][j] != 0 else 0.0

    def convolution2d(self):
        derfilt = np.array([1.0, -2, 1.0], dtype=np.float32)
        ck = signal.cspline2d(self.work_img, 8.0)
        self.work_img = (signal.sepfir2d(ck, derfilt, [1]) +
                 signal.sepfir2d(ck, [1], derfilt))

    def threshold(self):
        image = self.work_img
        thresh_min = threshold_mean(image)
        self.work_img = image > thresh_min

        # image = self.work_img
        # thresh_min = threshold_mean(image)
        # binary_min = image > thresh_min
        #
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        #
        # ax[0, 0].imshow(image, cmap=plt.cm.gray)
        # ax[0, 0].set_title('Original')
        #
        # ax[0, 1].hist(image.ravel(), bins=256)
        # ax[0, 1].set_title('Histogram')
        #
        # ax[1, 0].imshow(binary_min, cmap=plt.cm.gray)
        # ax[1, 0].set_title('Thresholded (min)')
        #
        # ax[1, 1].hist(image.ravel(), bins=256)
        # ax[1, 1].axvline(thresh_min, color='r')
        #
        # for a in ax[:, 0]:
        #     a.axis('off')
        # plt.show()

    def filter(self):
        self.pre_processing()
        self.segmentation()
        self.threshold()
        self.masking()
        # io.imsave('test1.jpg', self.work_img)

        # io.imsave('test.jpg', self.work_img)

        # fig, ax = try_all_threshold(self.work_img, figsize=(10, 8), verbose=True)
        # plt.show()

        # image = self.work_img
        # fig, ax = plt.subplots(ncols=2, subplot_kw={'adjustable': 'box'})
        #
        # ax[0].imshow(self.original_img)
        # ax[0].set_title('Original image')
        #
        # ax[1].imshow(self.work_img, cmap=plt.cm.gray)
        # ax[1].set_title('Frangi filter result')
        #
        # # ax[2].imshow(hessian(image), cmap=plt.cm.gray)
        # # ax[2].set_title('Hybrid Hessian filter result')
        #
        # for a in ax:
        #     a.axis('off')
        #
        # plt.tight_layout()
        # plt.show()

    def run(self):
        self.filter()
        self.result_img = self.work_img

    def compare_with_manual(self):
        white = 0
        black = 0
        white_count = 0
        black_count = 0
        for i in range(self.work_img.shape[0]):
            for j in range(self.work_img.shape[1]):
                manual = self.manual[i][j]
                mask = self.mask[i][j]
                img = self.work_img[i][j]
                if mask == 1:
                    if manual == 1:
                        black_count += 1
                        if img == 1:
                            black += 1
                    else:
                        white_count += 1
                        if img == 0:
                            white += 1
        print('white_count', white_count)
        print('white', white)
        print('white score', white/white_count)
        print('black_count', black_count)
        print('black', black)
        print('black score', black / black_count)
        print('score', (white + black) / (self.work_img.shape[0] * self.work_img.shape[1]))
