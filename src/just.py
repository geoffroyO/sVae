import lightfeaturesextract
import load_model as lm
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def load_data():
    print("... Loading data")
    spliced, copy_moved, spliced_mask, copy_moved_mask = lm.load_images("../data/CASIA2/Tp/", "../data/CASIA2/gt/")
    print("... Patching images")
    data, labels = lm.patch_images(spliced, spliced_mask)
    print("... Normalizing images")
    data = [rgb.astype('float32') / 255. for rgb in tqdm(data)]
    labels2 = []
    print("... Labelizing")
    for label in tqdm(labels):
        tp = np.sum(label) / 255
        percent = tp * 100 / (32 * 32)
        if 12.5 < percent < 87.5:
            labels2.append(1)
        else:
            labels2.append(0)
    labels = labels2
    tt = np.sum(labels)
    count = 0
    dataf = []
    labelsf = []
    for k, img in tqdm(enumerate(data)):
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

if __name__ == '__main__':
    data, labels = load_data()
    realData = []
    for k in range(len(labels)):
        if not labels[k]:
            realData.append(data[k])
    realData = np.array(realData)
    np.save("./data_to_load/oriSpliced.npy", realData)