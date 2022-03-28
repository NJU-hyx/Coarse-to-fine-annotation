from torch.utils.data import Dataset

import cv2
import os
import numpy as np

# import matplotlib.pyplot as plt


class dataset(Dataset):
    def __init__(self, root, hsilist, imagelist, labellist, groundtruth, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(dataset, self).__init__()
        self.root = root
        self.hsiList = hsilist
        self.imageList = imagelist
        self.labelList = labellist
        self.gt = groundtruth
        self.mean = mean
        self.std = std

    def __getitem__(self, item):
        # hsi = self.read_HSD(os.path.join(self.root, self.hsiList[item]))
        image = cv2.imread(os.path.join(self.root, self.imageList[item]), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root, self.labelList[item]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(self.root, self.gt[item]), cv2.IMREAD_GRAYSCALE)

        # image = image.transpose(1, 0, 2)[:, ::-1, :]  # hsicity dataset

        h, w = image.shape[0], image.shape[1]
        image = cv2.resize(image, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_NEAREST)
        # hsi = cv2.resize(hsi, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32)[:, :, ::-1] / 255.0

        # image = image - self.mean
        # image = image / self.std

        image = image.transpose((2, 0, 1))

        # return hsi.copy(), image.copy(), label.copy(), gt.copy()

        return image.copy(), label.copy(), gt.copy()

    def __len__(self):
        return len(self.imageList)

    def read_HSD(self, filename):
        # int32格式
        data = np.fromfile('%s' % filename, dtype=np.int32)
        height = data[0]
        width = data[1]
        SR = data[2]
        D = data[3]
        startw = data[4]  # 起始波段
        endw = data[6]  # 结束波段

        # float32格式
        data = np.fromfile('%s' % filename, dtype=np.float32)
        stepw = data[5]
        a = 7
        average = data[a:a + SR]
        a = a + SR
        coeff = data[a:a + D * SR].reshape((D, SR))
        a = a + D * SR
        scoredata = data[a:a + height * width * D].reshape((height * width, D))

        temp = np.dot(scoredata, coeff)

        data = (temp + average).reshape((height, width, SR))
        # data = (data - data.min()) / (data.max() - data.min())  # 全局归一化

        return data


