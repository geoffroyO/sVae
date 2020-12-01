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


if __name__ == '__main__':
    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    anodec = ano.load_anodec(dirFeatex, dirAno)

    model = postTreat(anodec)
    model.build(input_shape=(32, 32, 3))
    model.summary()
    """
    data = np.load("./data_to_load/spliced.npy")
    labels = np.load("./data_to_load/spliced_labels.npy")

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)
    """
