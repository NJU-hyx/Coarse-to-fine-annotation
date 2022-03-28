import numpy as np
import sklearn.neighbors
import scipy.sparse
import warnings
import imageio
import cv2
import time

nn = 10

# warnings.filterwarnings('ignore')


def knn_matte(img, trimap, mylambda=100):

    [m, n, c] = img.shape
    img, trimap = img, trimap / 255.0
    foreground = (trimap > 0.99).astype(int)
    background = (trimap < 0.01).astype(int)
    all_constraints = foreground + background

    print(time.asctime(time.localtime(time.time())) + ': Finding nearest neighbors')
    # a, b = np.unravel_index(np.arange(m * n), (m, n)).
    # feature_vec = np.append(np.transpose(img.reshape(m * n, c)), [a, b] / np.sqrt(m * m + n * n), axis=0).T

    # feature_vec = np.maximum(img.reshape(m * n, c), 0)
    feature_vec = img.reshape(m * n, c)
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=6).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    # Compute Sparse A
    print(time.asctime(time.localtime(time.time())) + ': Computing sparse A')
    row_inds = np.repeat(np.arange(m * n), 10)
    col_inds = knns.reshape(m * n * 10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1) / (c + 2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(m * n, m * n))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script - A
    D = scipy.sparse.diags(np.ravel(all_constraints))
    v = np.ravel(foreground)
    c = 2 * mylambda * np.transpose(v)
    H = 2 * (L + mylambda * D)

    print(time.asctime(time.localtime(time.time())) + ': Solving linear system for alpha')
    warnings.filterwarnings('error')

    try:
        # alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(m, n)
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.cg(H, c), 0), 1).reshape(m, n)

    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(m, n)

    return alpha


def main():
    root = 'F:/database/cityscapes/'
    img = cv2.imread(root + 'leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png', cv2.IMREAD_COLOR)
    map = cv2.imread(root + 'gtCoarse/train/aachen/aachen_000000_000019_gtCoarse_labelIds.png', cv2.IMREAD_GRAYSCALE)
    alpha = []

    for i in range(1, map.max()):
        if i in map:
            map_layer = (map == i).astype(int) * 255
            map_layer[map == 0, 1, 2, 3, 4, 5, 6] = 128
            alpha_layer = knn_matte(img, map_layer)
            alpha.append(alpha_layer)

    imageio.imwrite('donkeyAlpha.png', alpha)
    # plt.title('Alpha Matte')
    # plt.imshow(alpha)
    # plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.misc

    main()
