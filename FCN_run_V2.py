# -*- coding : utf-8-*-

import torch
from torch.nn import functional as F

from lib.models.vgg16 import vgg16
from lib.models.fcn8s import FCN8s
import lib.datasets as datasets

import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from knn_matting import knn_matte
from result_generation import result_generation

import timeit


def generate_code(data, code_length, R, pca):
    """
    Generate hashing code.

    Args
        data(torch.Tensor): Data.
        code_length(int): Hashing code length.
        R(torch.Tensor): Rotration matrix.
        pca(callable): PCA function.

    Returns
        pca_data(torch.Tensor): PCA data.
    """
    return (torch.from_numpy(pca.transform(data)).to(R.device) @ R).sign()


def get_confusion_matrix(label, pred, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    seg_pred = pred
    seg_gt = label.cpu().numpy()

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def itq_train(
        train_data,
        code_length,
        max_iter,
        device,
):
    """
    Training model.

    Args
        train_data(torch.Tensor): Training data.
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): Retrieval targets.
        code_length(int): Hash code length.
        max_iter(int): Number of iterations.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.

    Returns
        checkpoint(dict): Checkpoint.
    """
    train_data = (train_data - train_data.mean()) / train_data.std()

    # Initialization
    R = torch.randn((code_length, code_length), dtype=torch.float64).to(device)
    [U, _, _] = torch.svd(R)
    R = U[:, :code_length]

    # PCA
    pca = PCA(n_components=code_length)
    V = torch.from_numpy(pca.fit_transform(train_data)).to(device)

    # Training
    for i in range(max_iter):
        V_tilde = V @ R
        B = V_tilde.sign()
        [U, _, VT] = torch.svd(B.t() @ V)
        R = (VT.t() @ U.t())

    # Evaluate
    # Generate query code and retrieval code
    train_code = generate_code(train_data, code_length, R, pca).cpu().numpy()
    return train_code


def main():
    # model
    model = FCN8s()
    state_dict = torch.load('./parameters/fcn8s_from_caffe.pth')
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    # data
    # root = 'F:/database/cityscapes/'
    # imglist = ['leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png']
    # labellist = ['gtCoarse/train/aachen/aachen_000000_000019_gtCoarse_labelIds.png']

    # imglist = ['leftImg8bit/train/aachen/aachen_000045_000019_leftImg8bit.png']
    # labellist = ['gtCoarse/train/aachen/aachen_000045_000019_gtCoarse_labelIds.png']

    root = 'E:/NewDataset/HSIDatasetV2/'

    hsilist = [
               ]

    imglist = [r"val\rgb20210409_140722_03.jpg"
               ]
    labellist = [r"val\rgb20210409_140722_03_gray.png"
                 ]
    finelabellist = [r"test\rgb20210409_140722_03_gray.png"
                     ]

    dataset = datasets.dataset(
        root=root,
        hsilist=hsilist,
        imagelist=imglist,
        labellist=labellist,
        groundtruth=finelabellist
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    class LayerActivations:
        # features = None

        def __init__(self, model, layer_num):
            self.hook = model[layer_num].register_forward_hook(self.hook_fn)

        def hook_fn(self, module, input, output):
            self.features = output.cpu()

        def remove(self):
            self.hook.remove()

    conv_out = LayerActivations(model._modules, 'conv2_2')

    pca = PCA(n_components=1)
    pca3 = PCA(n_components=3)
    confusion_matrix = np.zeros((3, 19, 19))

    for k, (inputimage, inputlabel, gt) in enumerate(dataloader):

        h, w = inputimage.shape[2], inputimage.shape[3]

        inputimage = inputimage.cuda()
        with torch.no_grad():
            output = model.forward(inputimage)

        conv2_2_out = F.interpolate(
            input=conv_out.features, size=(h, w), mode='bilinear', align_corners=True)
        conv2_2_out = conv2_2_out.numpy().squeeze().transpose(1, 2, 0).reshape(-1, 128)
        conv2_2_out = (
                (conv2_2_out.T - np.mean(conv2_2_out, axis=1)) / np.std(conv2_2_out, axis=1)).T  # Standardization
        conv2_2_out_reduce = pca.fit_transform(conv2_2_out)  # F_i
        conv2_2_out_reduce /= conv2_2_out_reduce.max()

        # inputhsi = pca3.fit_transform(inputhsi.reshape(-1, 129))

        x, y = np.unravel_index(np.arange(h * w), (h, w)) / np.sqrt(h * h + w * w)
        x, y = x[:, np.newaxis], y[:, np.newaxis]

        img = inputimage.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 3)

        feature_map = np.concatenate((img, conv2_2_out_reduce, x, y), axis=1)  # X_i

        # feature_binary_code = itq_train(feature_map, 13, 32, 'cuda').reshape(h, w, -1)
        feature_binary_code = feature_map.reshape(h, w, -1)

        label_map = inputlabel.numpy()
        alpha = []
        classes = []

        for i in range(0, 34):
            if i in label_map:
                map_layer = (label_map == i).astype(np.uint8) * 255
                map_layer[label_map == 255] = 128
                alpha_layer = knn_matte(feature_binary_code, map_layer)
                cv2.imwrite(f'./result/{k}/img{i}.png', alpha_layer * 255)
                alpha.append(alpha_layer)
                classes.append(i)

        results = result_generation(alpha, classes, k, save=True)
        for i, result in enumerate(results):
            confusion_matrix[i] += get_confusion_matrix(gt[0], result, num_class=19, ignore=255)

    for i, thre in enumerate([0, 0.3, 0.7]):
        pos = confusion_matrix[i].sum(1)
        res = confusion_matrix[i].sum(0)
        tp = np.diag(confusion_matrix[i])
        pixel_acc = tp.sum() / pos.sum()
        acc_array = tp / np.maximum(1.0, pos)
        mean_acc = acc_array[1:].mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array[1:].mean()
        print(thre)
        print(f'MIou: {IoU_array}')
        print(mean_IoU)
        print(f'acc: {acc_array}')
        print(mean_acc)


if __name__ == '__main__':
    main()
