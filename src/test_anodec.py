from random import random

import anodec as ano

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, Dropout, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.losses.losses_impl import absolute_difference, Reduction

import numpy as np
from tqdm import tqdm


class postTreat(keras.Model):
    def __init__(self, anodec, **kwargs):
        super(postTreat, self).__init__(**kwargs)
        self.anodec = anodec
        self.subtract = Subtract()
        self.finalConv = Conv2D(1, 3, strides=1, padding='same', name='finalConv')

    def call(self, input):
        anoFeat = self.anodec(input)
        features = self.anodec.featex(input)
        sub = self.subtract([features, anoFeat])
        mask = self.finalConv(sub)
        return mask

def load_data():
    print("... Loading data")
    spliced, copy_moved, spliced_mask, copy_moved_mask = lm.load_images("../data/CASIA2/Tp/", "../data/CASIA2/gt/")
    print("... Patching images")
    data, labels = lm.patch_images(spliced, spliced_mask)
    print("... Normalizing images")
    data = [rgb.astype('float32') / 255. for rgb in tqdm(data)]
    labels2 = []
    print("... Labelizing")
    mask = labels
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
                labelsf.append(mask[k])
        if labels[k] == 1:
            dataf.append(img)
            labelsf.append(mask[k])
    data = dataf
    mask = labelsf
    tmp = list(zip(data, mask))
    random.shuffle(tmp)
    data, labels = zip(*tmp)
    data = np.array(data)
    mask = np.array(mask)
    return data, mask


if __name__ == '__main__':
    data, mask = load_data()
    np.save("./data_to_load/splicedFinal.npy", data)
    np.save("./data_to_laod/maskSplicedFinal.npy", mask)
    """
    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    anodec = ano.load_anodec(dirFeatex, dirAno)

    model = postTreat(anodec)
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    
    data = np.load("./data_to_load/spliced.npy")
    labels = np.load("./data_to_load/spliced_labels.npy")
    mask = np.load("./data_to

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)
    """
