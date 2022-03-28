import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

labels = [
    ('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    ('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    ('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    ('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    ('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    ('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    ('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    ('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    ('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    ('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    ('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    ('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    ('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    ('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    ('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    ('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    ('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    ('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    ('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    ('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    ('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    ('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    ('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    ('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    ('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    ('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    ('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    ('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    ('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    ('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    ('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    ('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    ('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    ('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    ('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

# hsicity_label = [
#     (0, (0, 0, 0)),  # background
#     (1, (0, 0, 142)),  # car
#     (2, (220, 20, 60)),  # human
#     (3, (128, 64, 128)),  # road
#     (4, (250, 170, 30)),  # traffic light
#     (5, (220, 220, 0)),  # traffic sign
#     (6, (107, 142, 35)),  # tree
#     (7, (70, 70, 70)),  # building
#     (8, (70, 130, 180)),  # sky
#     (9, (190, 153, 153)),  # object
# ]

hsicity_label = [
    (0, (128, 64, 128)),  # road
    (1, (244, 35, 232)),  # sidewalk
    (2, (70, 70, 70)),  # building
    (3, (102, 102, 156)),  # wall
    (4, (190, 153, 153)),  # fence
    (5, (153, 153, 153)),  # pole
    (6, (250, 170, 30)),  # traffic light
    (7, (220, 220, 0)),  # traffic sign
    (8, (107, 142, 35)),  # vegetation
    (9, (152, 251, 152)),  # terrain
    (10, (70, 130, 180)),  # sky
    (11, (220, 20, 60)),  # person
    (12, (255, 0, 0)),  # rider
    (13, (0, 0, 142)),  # car
    (14, (0, 0, 70)),  # truck
    (15, (0, 60, 100)),  # bus
    (16, (0, 80, 100)),  # train
    (17, (0, 0, 230)),  # motorcycle
    (18, (119, 11, 32)),  # bicycle
]


def get_hsicity_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, len(hsicity_label)):
        palette[j * 3] = hsicity_label[j][1][0]
        palette[j * 3 + 1] = hsicity_label[j][1][1]
        palette[j * 3 + 2] = hsicity_label[j][1][2]
    return palette


def get_cityscapes_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, len(labels)):
        palette[j * 3] = labels[j][7][0]
        palette[j * 3 + 1] = labels[j][7][1]
        palette[j * 3 + 2] = labels[j][7][2]
    return palette


def result_generation(stack, classes, k, save=False):
    label = np.asarray(stack)
    label_index = np.argmax(label, axis=0)

    result_output = []

    # threshold
    thresholdlist = [0, 0.3, 0.7]
    for threshold in thresholdlist:
        for m in range(0, label_index.shape[0]):
            for n in range(0, label_index.shape[1]):
                if label[label_index[m, n], m, n] <= threshold:
                    label_index[m, n] = -1

        # covert label
        result = label_index.copy()
        for j in range(len(classes)):
            result[label_index == j] = classes[j]
        result[label_index == -1] = 255

        # save result
        if save:
            # plt.imshow(result)
            if not os.path.exists(f'./result/{k}'):
                os.mkdir(f'./result/{k}')
            palette = get_hsicity_palette(256)
            save_label = Image.fromarray(result.astype(np.uint8))
            save_label.save(f'./result/{k}/result{threshold}_gray.png')
            save_label.putpalette(palette)
            save_label.save(f'./result/{k}/result{threshold}.png')
        result_output.append(result)

    return result_output


def main():
    stack = []
    classes = []
    for i in range(0, 34):  # cityscapes class
        label_map = cv2.imread(f'./result/6/img{i}.png', cv2.IMREAD_GRAYSCALE)
        if label_map is not None:
            stack.append(label_map)
            classes.append(i)
    label = np.asarray(stack) / 255
    label_index = np.argmax(label, axis=0)

    # threshold
    threshold_all = [0, 0.3, 0.7]
    for threshold in threshold_all:
        for m in range(0, label_index.shape[0]):
            for n in range(0, label_index.shape[1]):
                if label[label_index[m, n], m, n] <= threshold:
                    label_index[m, n] = -1

        # covert label
        result = label_index.copy()
        for j in range(len(classes)):
            result[label_index == j] = classes[j]
        result[label_index == -1] = 0

        # save result
        # plt.imshow(result)
        palette = get_hsicity_palette(256)
        save_label = Image.fromarray(result.astype(np.uint8))
        save_label.save(f'./result/result{threshold}_gray.png')
        save_label.putpalette(palette)
        save_label.save(f'./result/result{threshold}.png')


if __name__ == '__main__':
    main()