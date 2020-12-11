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


def preddiscrVae(model, img, block_size):
    N, M, C = img.shape
    reconstuction_img, features_img, mask_error, mask = np.zeros((N, M, C)), np.zeros((N, M, C)), \
                                                        np.zeros((N, M)), np.zeros((N, M))

    blocks = []
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            blocks.append(img[i:(i+block_size), j:(j+block_size)])

    blocks = np.array(blocks)
    features, reconstruction, error, mask_predict = model.predict(blocks)
    count = 0

    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            mask_pred = mask_predict[count]
            mask[i:(i+block_size), j:(j+block_size)] += mask_pred

            mask_error_pred = error[count]
            mask_error[i:(i+block_size), j:(j+block_size)] += mask_error_pred

            block_reconstruction = reconstruction[count]
            reconstuction_img[i:(i+block_size), j:(j+block_size)] += block_reconstruction

            block_features = features[count]
            features_img[i:(i+block_size), j:(j+block_size)] += block_features
            count += 1
    enum = enumMatrix(N, M, block_size)
    mask_error /= enum
    mask /= enum
    enum_3D = np.dstack((enum, enum))
    enum_3D = np.dstack((enum_3D, enum))
    reconstuction_img /= enum_3D
    features_img /= enum_3D
    return reconstuction_img, features_img, mask_error, mask


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

        reconstruction, features, error, mask = preddiscrVae(model, img, 32)
        np.save("./img_test/{}_reconstruction.npy".format(k), reconstruction)
        np.save("./img_test/{}_features.npy".format(k), features)
        np.save("./img_test/{}_error.npy".format(k), error)
        np.save("./img_test/{}_mask.npy".format(k), mask)

