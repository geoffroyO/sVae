import anodec as ano

from tensorflow.python.keras import Model

import numpy as np

if __name__ == '__main__':

    data = np.load("./data_to_load/splicedFinal.npy")
    mask = np.load("./data_to_load/maskSplicedFinal.npy")

    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"

    anodec = ano.load_anodec(dirFeatex, dirAno)
    model = Model(anodec.encoder.input, anodec.layers[8].output)
    model.summary()

    z_mean, z_log_var, z = anodec.encoder.predict(data[:5])

