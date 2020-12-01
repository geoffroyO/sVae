import anodec as ano
import lightfeaturesextract as lf

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.optimizers import Adam

import numpy as np

if __name__ == '__main__':
    data = np.load("./data_to_load/spliced.npy")
    labels = np.load("./data_to_load/spliced_labels.npy")

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    dir = "../pretrained_model/featex_spliced_250.h5"
    featex = lf.load_featex(dir)
    encoder = ano.encoder()
    decoder = ano.decoder()
    vae = ano.VAE(featex, encoder, decoder)

    vae.compile(optimizer=Adam(lr=1e-6))

    checkpoint = tf.keras.callbacks.ModelCheckpoint("../pretrained_model/anodec_spliced_250.h5",
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min')
    csv_logger = CSVLogger("anodec_spliced_250.csv", append=True)

    callbacks_list = [checkpoint, csv_logger]

    vae.fit(train_data, epochs=250, batch_size=128, validation_data=(test_data, test_data), callbacks=callbacks_list)