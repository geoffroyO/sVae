import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.losses.losses_impl import absolute_difference, Reduction

import numpy as np


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


def gaussian_blur(data, kernel_size=11, sigma=5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(data)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]
    data = tf.nn.depthwise_conv2d(data, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')
    return data


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


class srmAno(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(srmAno, self).__init__(**kwargs)
        self.srmConv2D = Conv2D(3, [5, 5], trainable=False, kernel_initializer=_build_SRM_kernel(),
                                activation=None, padding='same', strides=1,
                                bias_initializer=tf.constant_initializer(0.5))
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        blurred = gaussian_blur(inputs, kernel_size=3, sigma=5)
        noise_blurred = self.srmConv2D(blurred)

        features = self.srmConv2D(inputs)
        features = (features - noise_blurred) / 2 + 0.5
        _, _, z = self.encoder(features)
        reconstruction = self.decoder(z)
        L1 = absolute_difference(inputs, reconstruction, reduction=Reduction.NONE)
        error = tf.reduce_sum(L1, axis=-1)
        return features, reconstruction, error

    def train_step(self, data):
        if isinstance(data, tuple):
            mask = data[1]
            data = data[0]

        with tf.GradientTape() as tape:
            blurred = tf.stop_gradient(gaussian_blur(data, kernel_size=3, sigma=5))
            noise_blurred = self.srmConv2D(blurred)

            features = self.srmConv2D(data)
            features = (features - noise_blurred)/2 + 0.5

            z_mean, z_log_var, z = self.encoder(features)
            reconstruction = self.decoder(z)

            L1 = absolute_difference(features, reconstruction, reduction=Reduction.NONE)
            L1 = tf.math.multiply(mask, L1)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(L1, axis=[1, 2, 3]))

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            mask = data[1]
            data = data[0]
        blurred = gaussian_blur(data, kernel_size=3, sigma=5)
        noise_blurred = self.srmConv2D(blurred)

        features = self.srmConv2D(data)
        features = (features - noise_blurred) / 2 + 0.5

        z_mean, z_log_var, z = self.encoder(features)
        reconstruction = self.decoder(z)

        L1 = absolute_difference(features, reconstruction, reduction=Reduction.NONE)
        L1 = tf.math.multiply(mask, L1)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(L1, axis=[1, 2, 3]))

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
    data = np.load("./data_to_load/splicedBorderAndOri.npy")
    mask = np.load("./data_to_load/maskSplicedBorderAndOri.npy")
    train_data, test_data, train_mask, test_mask = train_test_split(data, mask, random_state=42)

    model = srmAno(encoder(), decoder())
    model.compile(optimizer=Adam(lr=1e-6))

    checkpoint = tf.keras.callbacks.ModelCheckpoint("../models/srmBlurredEndAno.h5",
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min')
    csv_logger = CSVLogger("srmBlurredEndAno_250.csv", append=True)

    callbacks_list = [checkpoint, csv_logger]

    model.fit(train_data, train_mask, epochs=250, batch_size=128,
              validation_data=(test_data, test_mask),
              callbacks=callbacks_list)
