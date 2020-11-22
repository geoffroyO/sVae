import dsvae as ds
import load_model as lm

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

import numpy as np
from tqdm import tqdm
import random


if __name__ == '__main__':

    dir = '../pretrained_model/model_1.h5'
    model = ds.load_dsvae(dir)
    svae_optimizer = Adam(lr=1e-6)
    pred_optimizer = Adam(lr=1e-6)
    model.compile(svae_optimizer, pred_optimizer, ds.predLoss, ds.vaeLoss)

    print("... Loading data and mask")

    spliced, copy_moved, spliced_mask, copy_moved_mask = lm.load_images("../data/CASIA2/Tp/", "../data/CASIA2/gt/")

    print("... Patching images and mask")
    data, mask = lm.patch_images(spliced[:800], spliced_mask[:800])

    print("... Normalizing images")
    data = np.array([rgb.astype('float32') / 255. for rgb in tqdm(data)])

    print("... Normalizing mask")
    mask = np.array([msk.astype('float32') / 255. for msk in tqdm(mask)])

    print("... Dealing with data and mask")

    tt = 0
    labels = []
    for index, msk in enumerate(tqdm(mask)):
        tp = np.sum(msk)
        percent = tp * 100 / (32 * 32)
        if not 10 < percent < 70:
            mask[index] = np.zeros((32, 32)).astype('float32')
            labels.append(0)
        else:
            tt += 1
            labels.append(1)

    print("... Balancing class")

    count = 0
    new_data, new_mask = [], []
    for k, img in enumerate(tqdm(data)):
        if labels[k] == 0:
            if count <= tt:
                count += 1
                new_data.append(img)
                new_mask.append(mask[k])
        if labels[k] == 1:
            new_data.append(img)
            new_mask.append(mask[k])

    data = np.array(new_data)
    mask = np.array(new_mask)

    tmp = list(zip(data, mask))
    random.shuffle(tmp)
    data, mask = zip(*tmp)

    train_data, test_data, train_mask, test_mask = train_test_split(data, mask, test_size=0.2, random_state=42)

    print("... Training")

    model.fit(np.array(train_data), np.array(train_mask), epochs=20, batch_size=128,
              validation_data=(np.array(test_data), np.array(test_mask)))
    model.save_weights("../pretrained_model/model.h5")

    mask_predict = model.predict(np.array(test_data))
    np.save("./mask_predict.npy", mask_predict)
    np.save("./test_mask.npy", test_mask)

