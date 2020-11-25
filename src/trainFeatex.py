import lightfeaturesextract
import load_model as lm
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def load_data():
    print("... Loading data")
    spliced, copy_moved, spliced_mask, copy_moved_mask = lm.load_images("../data/CASIA2/Tp/", "../data/CASIA2/gt/")
    print("... Patching images")
    data, labels = lm.patch_images(copy_moved[1500:], copy_moved_mask[1500:])
    print("... Normalizing images")
    data = [rgb.astype('float32') / 255. for rgb in tqdm(data)]
    labels2 = []
    print("... Labelizing")
    for label in tqdm(labels):
        tp = np.sum(label) / 255
        percent = tp * 100 / (32 * 32)
        if 10 < percent:
            labels2.append(1)
        else:
            labels2.append(0)
    labels = labels2
    tt = np.sum(labels)
    count = 0
    dataf = []
    labelsf = []
    for k, img in tqdm(enumerate(data)):
        if labels[k] == 0:
            if count <= tt:
                count += 1
                dataf.append(img)
                labelsf.append(labels[k])
        if labels[k] == 1:
            dataf.append(img)
            labelsf.append(labels[k])
    data = dataf
    labels = labelsf
    tmp = list(zip(data, labels))
    random.shuffle(tmp)
    data, labels = zip(*tmp)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def tmp():
    data = np.load("./data.npy", )
    labels = np.load("./labels.npy")

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = lightfeaturesextract.light_featex()
    optimizer = Adam(lr=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(),
                                                                            tf.keras.metrics.AUC(),
                                                                            tf.keras.metrics.Precision()])

    checkpoint = tf.keras.callbacks.ModelCheckpoint("../pretrained_model/featexAll250.h5",
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')
    csv_logger = CSVLogger("model_history_log.csv", append=True)

    callbacks_list = [checkpoint, csv_logger]

    history = model.fit(train_data, train_label, epochs=250, batch_size=128,
                        validation_data=(test_data, test_label), callbacks=callbacks_list)

    model.load_weights("../pretrained_model/featexAll250.h5")

    preds = model.predict(test_data, verbose=1)
    fpr, tpr, _ = roc_curve(test_label, preds)
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
    return None


if __name__ == '__main__':
    print("... Loading data")
    train_data = np.load("./train_data.npy")
    test_data = np.load("./test_data.npy")


    print("... Loading Labels")
    train_label = np.load("./train_label.npy")
    test_label = np.load("./test_label.npy")

    model = lightfeaturesextract.light_featex()
    optimizer = Adam(lr=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(),
                                                                            tf.keras.metrics.AUC(),
                                                                            tf.keras.metrics.Precision()])


    checkpoint = tf.keras.callbacks.ModelCheckpoint("../pretrained_model/featexAll80.h5",
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')
    csv_logger = CSVLogger("model_history_log.csv", append=True)

    callbacks_list = [checkpoint, csv_logger]

    history = model.fit(train_data, train_label, epochs=1, batch_size=128,
                        validation_data=(test_data, test_label), callbacks=callbacks_list)


    model.load_weights("../pretrained_model/featexAll80.h5")

    preds = model.predict(test_data, verbose=1)
    fpr, tpr, _ = roc_curve(test_label, preds)
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









