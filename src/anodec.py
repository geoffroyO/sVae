import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import random_normal
from tensorflow.python.keras.layers import Conv2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.losses.losses_impl import absolute_difference, Reduction

import numpy as np

def encoder():
    latent_dim = 2

    features_input = Input(shape=(28, 28, 1), name='features_in')
    encoder = Conv2D(32, 5, strides=2, padding="same")(features_input)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    encoder = Conv2D(64, 5, strides=2, padding="same")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    encoder = Conv2D(64, 1, padding='same')(encoder)

    encoder = Flatten()(encoder)

    z_mu = Dropout(0.25)(Dense(latent_dim, name="z_mu")(encoder))
    z_log_sigma = Dropout(0.25)(Dense(latent_dim, name="z_log_sigma")(encoder))

    z_sigma = tf.exp(z_log_sigma)
    z_vae = z_mu + random_normal(tf.shape(z_sigma)) * z_sigma
    encoder = Model(features_input, [z_mu, z_log_sigma, z_sigma, z_vae], name="encoder")

    return encoder


def decoder():
    latent_dim = 2
    decoder_input = Input(shape=(latent_dim,))

    decoder = Dropout(0.25)(Dense(7 * 7 * 64, activation="relu")(decoder_input))
    decoder = Reshape((7, 7, 64))(decoder)
    decoder = Conv2D(512, 1, padding='same')(decoder)

    decoder = BatchNormalization()(decoder)
    decoder = ReLU()(decoder)

    decoder = Conv2DTranspose(64, 2, strides=2, padding="same")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    decoder = Conv2DTranspose(32, 2, strides=2, padding="same")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    decoder = Conv2DTranspose(1, 5, activation='sigmoid', strides=1, padding='same')(decoder)
    decoder = Conv2D(1, 1, strides=1, padding='same', name='dec_conv2D_final', activation=tf.identity)(decoder)
    decoder = Model(decoder_input, decoder, name='decoder')

    return decoder


def vaeLoss(features, z_mu, z_sigma, reconstruction):
    kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.math.square(z_mu) +
                                                 tf.math.square(z_sigma) - tf.math.log(tf.square(z_sigma)) - 1,
                                                 axis=1))
    L1 = absolute_difference(features, reconstruction, reduction=Reduction.NONE)
    reconstrution_loss = tf.reduce_mean(tf.reduce_sum(L1, axis=[1, 2, 3]))
    return reconstrution_loss, kl_loss


class svae(Model):
    def __init__(self):
        super(svae, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def call(self, inputs):
        z_mean, z_log_sigma, z_sigma, z_vae = self.encoder(inputs)
        return self.decoder(z_vae)

    def compile(self, optimizer, loss, **kwargs):
        super(svae, self).compile()
        self.svae_optimizer = optimizer
        self.svaeLoss = loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mu, z_log_sigma, z_sigma, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstrution_loss, kl_loss = self.svaeLoss(data, z_mu, z_sigma, reconstruction)
            vae_loss = reconstrution_loss + kl_loss
        grads = tape.gradient(vae_loss, self.trainable_weights)
        self.svae_optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )
        return {
            "total_loss": vae_loss,
            "reconstruction_loss": reconstrution_loss,
            "kl_loss": kl_loss,
        }


if __name__ == '__main__':
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    def preprocess_images(images):
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')


    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_size = 60000
    batch_size = 32
    test_size = 10000

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size))
    sva = svae()
    optimizer = Adam(lr=1e-6)
    sva.compile(optimizer, vaeLoss)
    sva.fit(train_images, epochs=10, batch_size=32)