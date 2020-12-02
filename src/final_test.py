import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import final_training as ft
import anodec as ano


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
    for i in tqdm(range(N-block_size+1)):
        for j in range(M-block_size+1):
            blocks.append(img[i:(i+block_size), j:(j+block_size)])
    blocks = np.array(blocks)
    pred = model.predict(blocks)
    count = 0
    for i in tqdm(range(N-block_size+1)):
        for j in range(M-block_size+1):
            mask_pred = pred[count]
            mask[i:(i+block_size), j:(j+block_size)] += mask_pred[:, :, 0]
            count += 1
    enum = enumMatrix(N, M, block_size)
    mask /= enum
    return mask


if __name__=='__main__':
    path = "./img_test/1.jpg"
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.

    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    anodec = ano.load_anodec(dirFeatex, dirAno)

    model = ft.postTreat(anodec)
    model.predict(np.array([img[0:32,0:32]]))
    model.load_weights("../pretrained_model/final_250.h5")

    mask = pred(model, img, 32)
    plt.imsave("./test.jpg", arr=mask, format='jpg', color='gray')

