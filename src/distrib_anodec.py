import anodec as ano

from tensorflow.python.keras import Model

import numpy as np

if __name__ == '__main__':

    data = np.load("./data_to_load/splicedFinal.npy")
    labels = np.load("./data_to_load/spliced_labels.npy")
    data = data[:1500]
    labels = labels[:1500]

    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"

    anodec = ano.load_anodec(dirFeatex, dirAno)

    features_data = anodec.featex.predict(data)

    model = Model(anodec.encoder.input, anodec.encoder.layers[8].output)

    x = model.predict(features_data)

    z_mean, z_log_var, z = anodec.encoder.predict(features_data)

    np.load("./z_mean.npy", z_mean)
    np.load("./z_log_var.npy", z_log_var)