import lightfeaturesextract
import load_model as lm
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
import matplotlib.pyplot as plt



def load_data():
    spliced, copy_moved, spliced_mask, copy_moved_mask = lm.load_images("../data/CASIA2/Tp/", "../data/CASIA2/gt/")
    data, labels = lm.patch_images(spliced, spliced_mask)
    data = [rgb.astype('float32') / 255. for rgb in data]
    labels2 = []
    for label in labels:
        tp = np.sum(label) / 255
        percent = tp * 100 / (32 * 32)
        if 10 < percent < 70:
            labels2.append(1)
        else:
            labels2.append(0)
    labels = labels2
    tt = np.sum(labels)
    count = 0
    dataf = []
    labelsf = []
    for k, img in enumerate(data):
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


if __name__ == '__main__':
    data = np.load("./data.npy", )
    labels = np.load("./labels.npy")

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = lightfeaturesextract.light_featex()
    optimizer = Adam(lr=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint("../pretrained_model/featex.h5",
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')

    callbacks_list = [checkpoint]

    history = model.fit(train_data, train_label, epochs=3, batch_size=128,
                        validation_data=(test_data, test_label), callbacks=callbacks_list)

    model.load_weights("../pretrained_model/featex.h5")

    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./accuracy")


    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./loss")
    plt.close(fig)

    preds = model.predict(test_data, verbose=1)
    fpr, tpr, _ = roc_curve(test_data, preds)
    roc_auc = auc(fpr, tpr)
    print("***********")
    print(roc_auc)

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = {})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("./ROC")
    plt.close(fig)
