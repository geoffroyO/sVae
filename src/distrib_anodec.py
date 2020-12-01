import anodec as ano

from tensorflow.python.keras import Model

import numpy as np

if __name__ == '__main__':

    data = np.load("./data_to_load/splicedFinal.npy")
    labels = np.load("./data_to_load/spliced_labels.npy")
    data_tmp = []
    count_tamp, count_ori = 0, 0
    k = 0
    while count_tamp < 3 and count_ori < 3:
        if labels[k]:
            count_tamp += 1
            data_tmp.append(data[k])
        else:
            count_ori += 1
            data_tmp.append(data[k])
        k += 1

    data = data_tmp

    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"

    anodec = ano.load_anodec(dirFeatex, dirAno)

    features_data = anodec.featex.predict(data)
    print("****{}****".format(features_data.shape))

    model = Model(anodec.encoder.input, anodec.encoder.layers[8].output)

    x = model.predict(features_data)
    print("****{}****".format(x.shape))
    z_mean, z_log_var, z = anodec.encoder.predict(features_data)
    print("****{}****".format(z.shape))
    model = Model(anodec.decoder.input, anodec.decoder.layers[1].output)
    x_hat = model.predict(z)
    print("****{}****".format(x_hat.shape))


