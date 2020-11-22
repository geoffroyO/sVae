import lightfeaturesextract
import load_model as lm

from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
import cv2
from tqdm import tqdm


def training():
    data, labels = load_data()
    kf = KFold(n_splits=10)
    save_dir = '../pretrained_model/'
    fold_var = 1
    for train_index, val_index in kf.split(labels, data):
        training_data, training_labels = np.array(data[train_index]), np.array(labels[train_index])
        validation_data, validation_labels = np.array(data[val_index]), np.array(labels[val_index])

        model = lightfeaturesextract.light_featex()
        optimizer = Adam(lr=1e-6)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + 'model_' + str(fold_var) + '.h5',
                                                        monitor='val_accuracy', verbose=1,
                                                        save_best_only=True, mode='max')

        callbacks_list = [checkpoint]

        model.fit(training_data, training_labels, epochs=40, batch_size=128,
                  validation_data=(validation_data, validation_labels), callbacks=callbacks_list)

        tf.keras.backend.clear_session()

        fold_var += 1
    return None


def load_data():
    spliced, copy_moved, spliced_mask, copy_moved_mask = lm.load_images("../data/CASIA2/Tp/", "../data/CASIA2/gt/")
    data, labels = lm.patch_images(spliced[:10], spliced_mask[:10])
    data = [rgb.astype('float32') / 255. for rgb in data]
    labels2 = []
    for label in labels:
        tp = np.sum(label) / 255
        percent = tp * 100 / (32 * 32)
        if 10 < percent < 70:
            labels2.append(1)
        else:
            labels2.append(0)
    labels = labels2
    tt = np.sum(labels)
    count = 0
    dataf = []
    labelsf = []
    for k, img in enumerate(data):
        if labels[k] == 0:
            if count <= tt:
                count += 1
                dataf.append(img)
                labelsf.append(labels[k])
        if labels[k] == 1:
            dataf.append(img)
            labelsf.append(labels[k])
    data = dataf
    labels = labelsf
    tmp = list(zip(data, labels))
    random.shuffle(tmp)
    data, labels = zip(*tmp)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def enum_matrix(N, M, block_size, I, J):
    matrix = np.zeros((N, M)) + 1
    for i in I:
        for j in J:
            matrix[i:(i + block_size), j:(j + block_size)] += 1
    return matrix


def pred_map(model, image, block_size, I, J):
    N, M, _ = image.shape
    pred_map = np.zeros((N, M))
    for i in tqdm(I):
        for j in J:
            block = image[i:(i + block_size), j:(j + block_size)]
            label = model.predict(np.array([block]))[0, 0]
            pred_map[i:(i + block_size), j:(j + block_size)] += label
    enum_mat = enum_matrix(N, M, block_size, I, J)
    return pred_map / enum_mat


if __name__ == '__main__':
    J = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
                  104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
                  208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304,
                  312, 320, 328, 336, 344, 351])
    I = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
                  104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
                  208, 216, 221])
    img = cv2.imread("./test.jpg")
    img = img[..., ::-1]
    print(img.shape)
    dir = '../pretrained_model/model_1.h5'

    model = lightfeaturesextract.light_featex()
    model.load_weights(dir)
    optimizer = Adam(lr=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    pred = pred_map(model, img, 32, I, J)
    np.save("./pred_map.npy", pred)
