import dsvae as ds

from tensorflow.keras.optimizers import Adam

import numpy as np



if __name__ == '__main__':
    dir = '../pretrained_model/featex_test.h5'
    model = ds.load_dsvae(dir)
    vae_optimizer = Adam(lr=1e-6)
    model.compile(vae_optimizer, ds.vaeLoss)
    data = np.load("./spliced.npy")
    train, test = data[:int(len(data)*0.7)], data[int(len(data)*0.7):]
    print("... Training")

    model.fit(train, epochs=20, batch_size=128, validation=test)
    model.save_weights("../pretrained_model/model_test.h5")

