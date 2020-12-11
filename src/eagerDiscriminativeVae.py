import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras, square
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, Dropout, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_math_ops import squared_difference
from tensorflow.python.ops.losses.losses_impl import absolute_difference, Reduction

import numpy as np
import sys
from tensorflow.python.ops.math_ops import reduce_std


def _build_SRM_kernel():
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
    filters = np.einsum('klij->ijlk', filters)
    filters = filters.flatten()
    initializer_srm = tf.constant_initializer(filters)

    return initializer_srm


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def encoder():
    latent_dim = 128
    encoder_inputs = Input(shape=(32, 32, 3))

    x = Conv2D(32, 5, activation='relu', strides=2, padding="same")(encoder_inputs)
    x = BatchNormalization()(x)

    x = Conv2D(64, 5, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, 5, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    z_mean = Dropout(0.25)(Dense(latent_dim, activation='relu', name="z_mean")(x))
    z_log_var = Dropout(0.25)(Dense(latent_dim, activation='relu', name="z_log_var")(x))

    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def decoder():
    latent_inputs = keras.Input(shape=(128,))
    x = Dropout(0.25)(Dense(8 * 8 * 128, activation='relu')(latent_inputs))
    x = Reshape((8, 8, 128))(x)

    x = Conv2DTranspose(128, 1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(32, 3, strides=2, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    decoder_outputs = Conv2DTranspose(3, 3, activation='sigmoid', padding="same")(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def otsu(error):
    error = error.numpy()
    batch_size, n, m = error.shape
    sig_max, opti_tresh = np.zeros((batch_size)), np.zeros((batch_size))

    for eps in np.arange(0, 0.91, 0.01):
        for batch in range(batch_size):
            class_0, class_1 = [], []
            for i in range(n):
                for j in range(m):
                    err = error[batch, i, j]
                    if err < eps:
                        class_0.append(err)
                    else:
                        class_1.append(err)

            prob0, prob1 = len(class_0) / (n * m), len(class_1) / (n * m)

            if prob0 == 0 or prob1 == 0:
                sigma_w = 0

            else:
                mean0, mean1 = np.mean(class_0), np.mean(class_1)
                sigma_w = prob0*prob1*(mean0 - mean1)**2

            if sig_max[batch] < sigma_w:
                sig_max[batch] = sigma_w
                opti_tresh[batch] = eps
    return opti_tresh


def discriminative_labelling(error, treshold):
    thresh_ = treshold[..., tf.newaxis, tf.newaxis]
    out = error < thresh_
    mask = tf.cast(out, dtype=tf.int32)
    return mask


def dicriminative_error(error, threshold):
    error = error.numpy()
    batch_size, n, m = error.shape
    discr_err = np.zeros((batch_size))

    for batch in range(batch_size):
        class_0, class_1 = [], []
        for i in range(n):
            for j in range(m):
                if error[batch, i, j] < threshold[batch]:
                    class_0.append(error[batch, i, j])
                else:
                    class_1.append(error[batch, i, j])

        p0 = len(class_0)/(n*m)
        p1 = len(class_1)/(n*m)

        if p0 == 0:
            discr_err[batch_size] = 5

        elif p1 == 0:
            discr_err[batch_size] = np.mean(class_0) + 5

        else:
            sig_0 = np.std(class_0)**2
            sig_1 = np.std(class_1)**2
            sig = np.std(class_0 + class_1)**2
            discr_err[batch_size] = np.mean(class_0) + 5*(p0*sig_0 + p1*sig_1)/sig
    print(discr_err)
    return discr_err


class disciminativeAno(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(disciminativeAno, self).__init__(**kwargs)
        # self.srmConv2D = Conv2D(3, [5, 5], trainable=False, kernel_initializer=_build_SRM_kernel(),
        #                         activation=None, padding='same', strides=1)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        # srm_features = self.srmConv2D(inputs)
        srm_features = inputs
        _, _, z = self.encoder(srm_features)

        reconstruction = self.decoder(z)

        L2 = squared_difference(inputs, reconstruction)
        error = tf.reduce_mean(L2, axis=-1)

        treshold, _ = otsu(error)
        mask = discriminative_labelling(error, treshold)

        return mask

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # features = self.srmConv2D(data)
            features = data
            z_mean, z_log_var, z = self.encoder(features)
            reconstruction = self.decoder(z)

            L2 = squared_difference(features, reconstruction)
            error = tf.reduce_mean(L2, axis=-1)

            with tape.stop_recording():
                threshold = otsu(error)

            reconstruction_loss = dicriminative_error(error, threshold)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)

            kl_loss = -0.5*tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        features = self.srmConv2D(data)

        z_mean, z_log_var, z = self.encoder(features)
        reconstruction = self.decoder(z)

        L2 = squared_difference(features, reconstruction)
        error = tf.reduce_mean(L2, axis=-1)
        treshold, sigma_b = otsu(error)
        sigma, tau = reduce_std(error), 5

        discr_err = discriminative_labelling(error, treshold)

        reconstruction_loss = discr_err + tau * (1 - (sigma_b / sigma) ** 2)

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5

        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


if __name__ == '__main__':
    data = np.load("./data_to_load/spliced.npy")
    train_data, test_data = data[:int(len(data) * 0.7)], data[int(len(data) * 0.7):]

    model = disciminativeAno(encoder(), decoder())
    model.compile(optimizer=Adam(lr=1e-6), run_eagerly=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint("../pretrained_model/disciminativeAno.h5",
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min')
    csv_logger = CSVLogger("disciminativeAno_spliced_250.csv", append=True)

    callbacks_list = [checkpoint, csv_logger]

    model.fit(train_data, epochs=250, batch_size=128) # , validation_data=(test_data, test_data), callbacks=callbacks_list
