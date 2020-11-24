import lightfeaturesextract

from tensorflow.keras.optimizers import Adam

from tqdm import tqdm
import numpy as np
import cv2


def enum_matrix(N, M, block_size):
    matrix = np.zeros((N, M))
    for i in range(N-block_size + 1):
        for j in range(M-block_size + 1):
            matrix[i:(i + block_size), j:(j + block_size)] += 1
    return matrix


def pred_map(model, image, block_size):
    N, M, _ = image.shape
    pred_map = np.zeros((N, M))
    blocks = []
    for i in tqdm(range(N-block_size+1)):
        for j in range(M-block_size+1):
            block = image[i:(i + block_size), j:(j + block_size)]
            blocks.append(block)
    labels = model.predict(np.array(blocks))
    count = 0
    for i in tqdm(range(N-block_size+1)):
        for j in range(M-block_size+1):
            pred_map[i:(i + block_size), j:(j + block_size)] += labels[count][0]
            count += 1
    enum_mat = enum_matrix(N, M, block_size)
    return pred_map / enum_mat


if __name__ == '__main__':
    model = lightfeaturesextract.light_featex()
    optimizer = Adam(lr=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.load_weights("../pretrained_model/featex.h5")

    img = cv2.imread("./test.tif", 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.

    pred = pred_map(model, img, 32)
    np.save("./prediction.npy", pred)
