import dsvae as ds
import load_model as lm

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tqdm import tqdm
import random


if __name__ == '__main__':

    dir = "../pretrained_model/featex_test.h5"
    model = ds.load_dsvae(dir)
    model.featex.summary()
    data = np.load("./spliced.npy")

    optimizer = Adam(learning_rate=1e-3)
    mse_loss_fn = MeanSquaredError()

    loss_metric = Mean()

    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    print(train_dataset.shape)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    print(train_dataset.shape)

    epochs = 2

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(tqdm(train_dataset)):
            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_train)
                # Compute reconstruction loss
                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(model.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
    print("... Training")

