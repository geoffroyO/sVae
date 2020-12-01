import anodec as ano

from tensorflow.python.keras import Model

import numpy as np

if __name__ == '__main__':

    data = np.load("./data_to_load/splicedFinal.npy")
    mask = np.load("./data_to_load/maskSplicedFinal.npy")

    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"

    anodec = ano.load_anodec(dirFeatex, dirAno)

    features_data = anodec.featex.predict(data[:5])
    print("****{}****".format(features_data.shape))
    model = Model(anodec.encoder.input, anodec.encoder.layers[8].output)

    x = model.predict(features_data[:5])
    z_mean, z_log_var, z = anodec.encoder.predict(features_data[:5])

