import discriminativeVae as dv
import cv2
import numpy as np
from tqdm import tqdm


def enumMatrix(N, M, block_size):
    enum = np.zeros((N, M))
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            enum[i:(i+block_size), j:(j+block_size)] += 1
    return enum


def predColumbia1(model, img, block_size):
    N, M, C = img.shape
    blocks = []
    for i in range(N - block_size + 1):
        for j in range(M - block_size + 1):
            blocks.append(img[i:(i + block_size), j:(j + block_size)])

    blocks = np.array(blocks)
    features, reconstruction = model.predict(blocks)
    np.save("./img_test/featuresCo.npy", features)
    np.save("./img_test/reconstructionCo.npy", reconstruction)
    return None


def predColumbia2(img, block_size):
    N, M, C = img.shape
    reconstuction_img, features_img = np.zeros((N, M, C)), np.zeros((N, M, C))

    enum = enumMatrix(N, M, block_size)
    enum_3D = np.dstack((enum, enum))
    enum_3D = np.dstack((enum_3D, enum))

    features = np.load("./img_test/featuresCo.npy")
    count = 0
    for i in range(N - block_size + 1):
        for j in range(M - block_size + 1):
            block_features = features[count]
            features_img[i:(i + block_size), j:(j + block_size)] += block_features
            count += 1
    features_img /= enum_3D
    np.save("./img_test/9_features.npy", features)

    reconstruction = np.load("./img_test/reconstructionCo.npy")
    count = 0
    for i in range(N - block_size + 1):
        for j in range(M - block_size + 1):
            block_reconstruction = reconstruction[count]
            reconstuction_img[i:(i + block_size), j:(j + block_size)] += block_reconstruction
            count += 1
    reconstuction_img /= enum_3D
    np.save("./img_test/9_reconstruction.npy", reconstuction_img)

    return None


def preddiscrVae(model, img, block_size):
    N, M, C = img.shape
    reconstuction_img, features_img = np.zeros((N, M, C)), np.zeros((N, M, C))

    blocks = []
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            blocks.append(img[i:(i+block_size), j:(j+block_size)])

    blocks = np.array(blocks)
    features, reconstruction = model.predict(blocks[:int(len(blocks)/2)], batch_size=128)
    features2, reconstruction2 = model.predict(blocks[int(len(blocks)/2):], batch_size=128)
    features += features2
    reconstruction += reconstruction2
    count = 0

    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            block_reconstruction = reconstruction[count]
            reconstuction_img[i:(i+block_size), j:(j+block_size)] += block_reconstruction

            block_features = features[count]
            features_img[i:(i+block_size), j:(j+block_size)] += block_features
            count += 1

    enum = enumMatrix(N, M, block_size)
    enum_3D = np.dstack((enum, enum))
    enum_3D = np.dstack((enum_3D, enum))
    reconstuction_img /= enum_3D
    features_img /= enum_3D
    return reconstuction_img, features_img


if __name__=='__main__':
    pathModel = "../models/discriminativeAno_30.h5"

    encoder = dv.encoder()
    decoder = dv.decoder()
    model = dv.discriminativeAno(encoder, decoder)

    path = "./img_test/{}.jpg".format(1)
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.
    model.predict(np.array([img[0:32, 0:32]]))

    model.load_weights(pathModel)

    for k in tqdm(range(1, 9)):
        if k == 8 or k == 9:
            path = "./img_test/{}.tif".format(k)
        else:
            path = "./img_test/{}.jpg".format(k)

        img = cv2.imread(path, 1)
        img = img[..., ::-1]
        img = img.astype('float32') / 255.

        reconstruction, features = preddiscrVae(model, img, 32)
        np.save("./img_test/{}_reconstruction.npy".format(k), reconstruction)
        np.save("./img_test/{}_features.npy".format(k), features)

