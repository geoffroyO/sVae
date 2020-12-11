import discriminativeVae as dv
import cv2
import numpy as np


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
            mask[i:(i+block_size), j:(j+block_size)] += mask_pred[0]
            count += 1
    enum = enumMatrix(N, M, block_size)
    mask /= enum
    return mask


if __name__=='__main__':
    pathModel = "../pretrained_model/disciminativeAno_20.h5"

    encoder = dv.encoder()
    decoder = dv.decoder()
    model = dv.disciminativeAno(encoder, decoder)
    path = "./img_test/{}.jpg".format(1)
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.
    model.predict(np.array([img[0:32, 0:32]]))

    model.load_weights(pathModel)

    for k in tqdm(range(1, 9)):
        if k == 8:
            path = "./img_test/{}.tif".format(k)
        else:
            path = "./img_test/{}.jpg".format(k)
        img = cv2.imread(path, 1)
        img = img[..., ::-1]
        img = img.astype('float32') / 255.

        reconstruction, features, error = predendVae(model, img, 32)
