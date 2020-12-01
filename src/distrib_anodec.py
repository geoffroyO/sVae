import anodec as ano

import numpy as np

if __name__ == '__main__':

    data = np.load("./data_to_load/splicedFinal.npy")
    mask = np.load("./data_to_load/maskSplicedFinal.npy")

    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    anodec = ano.load_anodec(dirFeatex, dirAno)
    print("****{}****".format(len(anodec.layers)))

    z_mean, z_log_var, z = anodec.predict(data[:5])

