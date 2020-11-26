import dsvae as ds

from tensorflow.keras.optimizers import Adam

import numpy as np
from tensorflow.python.keras.optimizers import sgd

if __name__ == '__main__':
    dir = '../pretrained_model/featex_test.h5'
    model = ds.load_dsvae(dir)
    vae_optimizer = sgd(lr=1e-6)
    model.compile(vae_optimizer, ds.vaeLoss)
    data = np.load("./spliced.npy")
    train = data[:128]
    print("... Training")

    model.fit(train, epochs=2000, batch_size=32)
    model.save_weights("../pretrained_model/model_test.h5")

