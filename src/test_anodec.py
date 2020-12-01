import random

import anodec as ano
import load_model as lm

import tensorflow as tf
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
        self.batchNorm = BatchNormalization()
        self.subtract = Subtract()
        self.finalConv = Conv2D(1, 3, padding='same', activation='sigmoid', name='finalConv')

    def call(self, input):
        features = self.anodec.featex(input)
        anoFeat = self.anodec(input)

        sub = self.subtract([features, anoFeat])
        sub = self.batchNorm(sub)
        mask = self.finalConv(sub)
        return mask


def dice(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()
    mean_dice = 0
    batch_size, N, M = img1.shape
    for k in range(batch_size):
        inter = 0
        tt1, tt2 = 0, 0
        for i in range(N):
            for j in range(M):
                ind1, ind2 = img1[i, j], img2[i, j]
                if ind1:
                    tt1 += 1
                if ind2:
                    tt2 += 1
                if ind1 and ind2:
                    inter += 1
                dice = 2 * inter / (tt1 + tt2)
        mean_dice += dice
    return mean_dice/batch_size


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)

    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    anodec = ano.load_anodec(dirFeatex, dirAno)

    model = postTreat(anodec)
    model.compile(loss='mse', optimizer=Adam(lr=1e-6), metrics=[dice], run_eagerly=True)

    data = np.load("./data_to_load/splicedFinal.npy")
    mask = np.load("./data_to_load/maskSplicedFinal.npy")


    train_data, test_data, train_mask, test_mask = train_test_split(data, mask, test_size=0.2, random_state=42)
    model.fit(train_data, train_mask, epochs=10, validation_data=(test_data, test_mask), batch_size=128)
