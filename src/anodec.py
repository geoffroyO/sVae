import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import random_normal
from tensorflow.python.keras.layers import Conv2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, ReLU


def encoder():
    latent_dim = 128

    features_input = Input(shape=(32, 32, 128), name='features_in')
    encoder = Conv2D(256, 5, strides=2, padding="same")(features_input)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    encoder = Conv2D(512, 5, strides=2, padding="same")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    encoder = Conv2D(512, 1, padding='same')(encoder)

    encoder = Flatten()(encoder)

    z_mu = Dropout(0.25)(Dense(latent_dim, name="z_mu")(encoder))
    z_log_sigma = Dropout(0.25)(Dense(latent_dim, name="z_log_sigma")(encoder))

    z_sigma = tf.exp(z_log_sigma)
    z_vae = z_mu + random_normal(tf.shape(z_sigma)) * z_sigma
    encoder = Model(features_input, [z_mu, z_log_sigma, z_sigma, z_vae], name="encoder")

    return encoder


def decoder():
    latent_dim = 128
    decoder_input = Input(shape=(latent_dim,))

    decoder = Dropout(0.25)(Dense(8 * 8 * 512, activation="relu")(decoder_input))
    decoder = Reshape((8, 8, 512))(decoder)
    decoder = Conv2D(512, 1, padding='same')(decoder)

    decoder = BatchNormalization()(decoder)
    decoder = ReLU()(decoder)

    decoder = Conv2DTranspose(512, 2, strides=2, padding="same")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    decoder = Conv2DTranspose(256, 2, strides=2, padding="same")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    decoder = Conv2DTranspose(128, 5, activation='sigmoid', strides=1, padding='same')(decoder)
    decoder = Conv2D(128, 1, strides=1, padding='same', name='dec_conv2D_final', activation=tf.identity)(decoder)
    decoder = Model(decoder_input, decoder, name='decoder')

    return decoder


class svae(Model):
    def __init__(self):
        super(svae, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def call(self, inputs):
        z_mean, z_log_var, z_var, z_vae = self.encoder(inputs)
        return self.decoder(z_vae)


if __name__ == '__main__':
    model = decoder()
    model.build(input_shape=(None, 128))
    model.summary()
