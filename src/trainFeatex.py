import lightfeaturesextract
import load_model as lm

from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
import cv2
from tqdm import tqdm

def load_data():
    spliced, copy_moved, spliced_mask, copy_moved_mask = lm.load_images("../data/CASIA2/Tp/", "../data/CASIA2/gt/")
    data, labels = lm.patch_images(spliced, spliced_mask)
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


if __name__=='__main__':
    data, labels = load_data()
    data, labels = np.array(data), np.array(labels)
    np.save("./data.npy", data)
    np.save("./labels.npy", labels)