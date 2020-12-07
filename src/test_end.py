import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn

import anodec as ano
import final_training as ft


def enumMatrix(N, M, block_size):
    enum = np.zeros((N, M))
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            enum[i:(i+block_size), j:(j+block_size)] += 1
    return enum


def pred(model, img, block_size):
    N, M, _ = img.shape
    mask = np.zeros((N, M))
    blocks = []
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            blocks.append(img[i:(i+block_size), j:(j+block_size)])
    blocks = np.array(blocks)
    pred = model.predict(blocks)
    count = 0
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            mask_pred = pred[count]
            mask[i:(i+block_size), j:(j+block_size)] += mask_pred
            count += 1
    enum = enumMatrix(N, M, block_size)
    mask /= enum
    return mask


if __name__ == '__main__':
    path_featex = "../pretrained_model/new_featex_250.h5"
    path_anodec = "../pretrained_model/new_anodec_250.h5"
    anodec = ano.load_anodec(path_featex, path_anodec)

    model = ft.postTreat(anodec)

    path = "./img_test/{}.jpg".format(1)
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.

    res = model.predict(np.array([img[0:+32, 0:32], img[0:+32, 0:32], img[0:+32, 0:32]]))
    print("****{}***".format(res.shape))
    """
    for k in tqdm(range(1, 7)):
        path = "./img_test/{}.jpg".format(k)
        img = cv2.imread(path, 1)
        img = img[..., ::-1]
        img = img.astype('float32') / 255.

        mask = pred(model, img, 32)
        figure = plt.figure()
        sn.heatmap(mask, cmap="YlGnBu", center=np.mean(mask))
        plt.imsave("./img_test/{}_pred_gt.jpg".format(k), arr=mask, format='jpg')
        plt.close(figure)
    """