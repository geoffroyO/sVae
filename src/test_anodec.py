import anodec as ano
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == '__main__':
    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    model = ano.load_anodec(dirFeatex, dirAno)

    data = np.load("./data_to_load/spliced.npy")
    labels = np.load("./data_to_load/spliced_labels.npy")

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)
    model.evaluate(test_data, test_data, batch_size=128)
