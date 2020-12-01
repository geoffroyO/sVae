import lightfeaturesextract as lf

import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.losses.losses_impl import absolute_difference, Reduction

import numpy as np



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
    encoder_inputs = Input(shape=(32, 32, 128))

    x = Conv2D(256, 5, strides=2, padding="same")(encoder_inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 1, padding='same')(x)

    x = Flatten()(x)

    z_mean = Dropout(0.25)(Dense(latent_dim, name="z_mean")(x))
    z_log_var = Dropout(0.25)(Dense(latent_dim, name="z_log_var")(x))

    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def decoder():
    latent_inputs = keras.Input(shape=(128,))
    x = Dropout(0.25)(Dense(8 * 8 * 512)(latent_inputs))
    x = Reshape((8, 8, 512))(x)
    x = Conv2D(512, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(512, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    decoder_outputs = Conv2D(128, 1, strides=1,  padding='same', activation='sigmoid')(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


class VAE(keras.Model):
    def __init__(self, featex, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.featex = featex
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        features = self.featex(inputs)
        _, _, z = self.encoder(features)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            features = self.featex(data)
            z_mean, z_log_var, z = self.encoder(features)
            reconstruction = self.decoder(z)
            L1 = absolute_difference(features, reconstruction, reduction=Reduction.NONE)
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
            data = data[0]
        features = self.featex(data)
        z_mean, z_log_var, z = self.encoder(features)
        reconstruction = self.decoder(z)
        L1 = absolute_difference(features, reconstruction, reduction=Reduction.NONE)
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


def load_anodec(dirFeatex, dirAno):
    featex = lf.load_featex(dirFeatex)

    anodec = VAE(featex, encoder(), decoder())
    anodec.compile(optimizer=Adam(lr=1e-6))

    data = np.load("./data_to_load/spliced.npy")
    anodec.predict(data[:1])

    anodec.load_weights(dirAno)
    anodec.trainable = False
    return anodec