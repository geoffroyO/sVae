import lightfeaturesextract
import load_model as lm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_data():
    print("... Load images")
    spliced, copy_moved, spliced_mask, copy_moved_mask = lm.load_images("../data/CASIA2/Tp/", "../data/CASIA2/gt/")
    print("... Patch images")
    data, labels = lm.patch_images(spliced, spliced_mask)
    print("... Normalization")
    data = [rgb.astype('float32') / 255. for rgb in tqdm(data)]
    labels2 = []
    print("... Labeling")
    for label in tqdm(labels):
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
    print("... Balancing class")
    for k, img in enumerate(tqdm(data)):
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


if __name__=="__main__":
    data, labels = load_data()

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_data, test_data = np.array(train_data), np.array(test_data)
    train_labels, test_labels = np.array(train_labels), np.array(test_labels)

    model = lightfeaturesextract.light_featex()
    optimizer = Adam(lr=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../pretrained_model/model_featex.h5',
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')

    callbacks_list = [checkpoint]

    model.fit(train_data, train_labels, epochs=40, batch_size=128,
              validation_data=(test_data, test_labels), callbacks=callbacks_list)

    model.load_weights('../pretrained_model/model_featex.h5')
    preds = model.predict(test_data)

    fpr, tpr, _ = roc_curve(test_labels, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
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
    plt.savefig("./testfig")
