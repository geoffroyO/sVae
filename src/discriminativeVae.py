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
    sig_max, opti_tresh = tf.reduce_sum(tf.zeros_like(error), axis=[1, 2]), \
                          tf.reduce_sum(tf.zeros_like(error), axis=[1, 2])

    for eps in np.arange(0.1, 0.91, 0.01):

        cond0 = tf.where(error < eps, error, tf.zeros_like(error))
        cond1 = tf.where(error >= eps, error, tf.zeros_like(error))

        N0 = tf.math.count_nonzero(cond0, axis=[1, 2], dtype=tf.float32)
        N1 = tf.math.count_nonzero(cond1, axis=[1, 2], dtype=tf.float32)

        mean0 = tf.reduce_mean(cond0, axis=[1, 2])*((N1+N0)/N0)
        mean0 = tf.where(tf.math.is_nan(mean0), tf.zeros_like(mean0), mean0)
        mean1 = tf.reduce_mean(cond1, axis=[1, 2])*((N1+N0)/N1)
        mean1 = tf.where(tf.math.is_nan(mean1), tf.zeros_like(mean1), mean1)


        prob0 = N0/(N1+N0)
        prob1 = N1/(N1+N0)
        print(prob0)
        prob0.append(3)

        sig_b = prob0*prob1*(mean0-mean1)**2
        bool = tf.math.greater_equal(sig_b, sig_max)
        sig_max = tf.where(bool, sig_b, sig_max)
        opti_tresh = tf.where(bool, tf.zeros_like(opti_tresh)+eps, opti_tresh)

    return opti_tresh


def discriminative_labelling(error, treshold):
    tresh_ = treshold[..., tf.newaxis, tf.newaxis]
    out = error < tresh_
    mask = tf.cast(out, dtype=tf.int32)
    return mask


def dicriminative_error(error, threshold):
    thresh_ = threshold[..., tf.newaxis, tf.newaxis]
    out = error < thresh_

    mask1 = tf.cast(out, dtype=tf.float32)
    mask2 = 1 - mask1

    error1 = tf.math.multiply(error, mask1)
    error2 = tf.math.multiply(error, mask2)

    N1 = tf.reduce_sum(mask1, axis=[1, 2])
    N2 = tf.reduce_sum(mask2, axis=[1, 2])


    prob1 = tf.reduce_mean(mask1, axis=[1, 2])
    prob2 = tf.reduce_mean(mask2, axis=[1, 2])

    mean1 = tf.math.divide(tf.reduce_sum(error1, axis=[1, 2]), N1)
    mean2 = tf.math.divide(tf.reduce_sum(error2, axis=[1, 2]), N2)

    sigmab = prob1*prob2*(mean1-mean2)**2

    return mean1, sigmab


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
                print(threshold)
                threshold += 2

            sigma = reduce_std(error, axis=[1, 2])
            discr_err, sigma_b = dicriminative_error(error, threshold)

            reconstruction_loss = discr_err + 5 * (1 - (sigma_b / sigma) ** 2)
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
