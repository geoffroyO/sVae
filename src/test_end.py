import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import endVae as ev

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
            mask[i:(i+block_size), j:(j+block_size)] += mask_pred[:, :, 0]
            count += 1
    enum = enumMatrix(N, M, block_size)
    mask /= enum
    return mask


if __name__ == '__main__':
    path = "./img_test/1.jpg"
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.


    encoder = ev.encoder()
    decoder = ev.decoder()

    model = ev.srmAno(encoder, decoder)
    model.predict(np.array([img[0:32, 0:32]]))
    model.load_weights("../pretrained_model/srmAno.h5")

    for k in tqdm(range(1, 7)):
        path = "./img_test/{}.jpg".format(k)
        img = cv2.imread(path, 1)
        img = img[..., ::-1]
        img = img.astype('float32') / 255.

        mask = pred(model, img, 32)
        N, M = mask.shape
        for i in range(N):
            for j in range(M):
                if mask[i, j] > 0.6:
                    mask[i, j] = 255
                else:
                    mask[i, j] = 0
        plt.imsave("./img_test/{}_pred_gt.jpg".format(k), arr=mask, format='jpg', cmap="gray")