import anodec as ano

import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow import keras

from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Subtract
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt


class postTreat(keras.Model):
    def __init__(self, anodec, **kwargs):
        super(postTreat, self).__init__(**kwargs)
        self.anodec = anodec
        self.batchNorm = BatchNormalization()
        self.subtract = Subtract()
        self.finalConv = Conv2D(1, 3, padding='same', activation='sigmoid', name='finalConv')

    def call(self, input):
        features = self.anodec.featex(input)
        anoFeat = self.anodec(input)

        squareFeat = tf.math.square(features)
        squareanoFeat = tf.math.square(anoFeat)

        sumFeat = tf.reduce_sum(squareFeat, axis=-1)
        sumanoFeat = tf.reduce_sum(squareanoFeat, axis=-1)

        sqrtFeat = tf.math.sqrt(sumFeat)
        sqrtanoFeat = tf.math.sqrt(sumanoFeat)

        sub = self.subtract([sqrtFeat, sqrtanoFeat])
        sub = self.batchNorm(sub)
        mask = self.finalConv(sub)
        return mask


def dice(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (tf.reduce_sum(tf.square(y_true), -1)
                                           + tf.reduce_sum(tf.square(y_pred), -1) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)


if __name__ == '__main__':
    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    anodec = ano.load_anodec(dirFeatex, dirAno)

    model = postTreat(anodec)
    model.build(input_shape=(32,32,3))
    model.summary()

if __name__ == '__main__':
    data = np.load("./data_to_load/splicedFinal.npy")
    mask = np.load("./data_to_load/maskSplicedFinal.npy")

    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    anodec = ano.load_anodec(dirFeatex, dirAno)

    model = postTreat(anodec)
    model.compile(loss=dice_loss, optimizer=Adam(lr=1e-6), metrics=[dice, tf.keras.metrics.Recall(),
                                                                                tf.keras.metrics.AUC(),
                                                                                tf.keras.metrics.Precision()])

    checkpoint = tf.keras.callbacks.ModelCheckpoint("../pretrained_model/final_250.h5",
                                                    monitor='val_dice', verbose=1,
                                                    save_best_only=True, mode='max')
    csv_logger = CSVLogger("model_history_final_250.csv", append=True)

    callbacks_list = [checkpoint, csv_logger]

    train_data, test_data, train_mask, test_mask = train_test_split(data, mask, test_size=0.2, random_state=42)
    model.fit(train_data, train_mask, epochs=250, validation_data=(test_data, test_mask),
              batch_size=128, callbacks=callbacks_list)

    model.load_weights("../pretrained_model/final_250.h5")

    preds = model.predict(test_data, verbose=1)
    fpr, tpr, _ = roc_curve(test_mask, preds)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("./ROC")
    plt.close(fig)
