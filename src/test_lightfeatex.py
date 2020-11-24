import lightfeaturesextract

from tensorflow.keras.optimizers import Adam

import numpy as np


if __name__ == '__main__':
    data = np.load("./data.npy", )
    labels = np.load("./labels.npy")

    model = lightfeaturesextract.light_featex()
    optimizer = Adam(lr=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.load_weights("../pretrained_model/featex.h5")

    res = model.predict(np.array([data[0]]), verbose=1)
    print(res)