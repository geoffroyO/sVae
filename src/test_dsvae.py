import dsvae as ds
import load_model as lm

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

import numpy as np
from tqdm import tqdm
import random


if __name__ == '__main__':

    dir = '../pretrained_model/model_1.h5'
    model = ds.load_dsvae(dir)
    vae_optimizer = Adam(lr=1e-6)
    model.compile(vae_optimizer, ds.vaeLoss)
    data = np.load("./spliced.npy")

    print("... Training")

    model.fit(data, epochs=20, batch_size=128)
    model.save_weights("../pretrained_model/model.h5")

