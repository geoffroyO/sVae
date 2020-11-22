import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import random_normal
from tensorflow.python.keras.layers import Conv2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, ReLU


def encoder():
    latent_dim = 2

    features_input = Input(shape=(32, 32, 128), name='features_in')
    encoder = Conv2D(256, 5, strides=2, padding="same")(features_input)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    encoder = Conv2D(512, 5, strides=2, padding="same")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    encoder = Dropout(0.25)(encoder)
    encoder = Flatten()(encoder)

    z_mean = Dropout(0.25)(Dense(latent_dim, name="z_mean")(encoder))
    z_log_var = Dropout(0.25)(Dense(latent_dim, name="z_log_var")(encoder))

    z_var = tf.exp(z_log_var)
    z_vae = z_mean + random_normal(tf.shape(z_var)) * z_var
    encoder = Model(features_input, [z_mean, z_log_var, z_var, z_vae], name="encoder")

    return encoder


def decoder():
    latent_dim = 2
    decoder_input = Input(shape=(latent_dim,))

    decoder = Dense(8 * 8 * 512, activation="relu")(decoder_input)
    decoder = Reshape((8, 8, 512))(decoder)

    decoder = BatchNormalization()(decoder)
    decoder = ReLU()(decoder)

    decoder = Conv2DTranspose(512, 2, strides=2, padding="same")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    decoder = Conv2DTranspose(256, 2, strides=2, padding="same")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    decoder = Conv2DTranspose(128, 5, activation='sigmoid', strides=1, padding='same')(decoder)
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
    model = svae()
    model.build(input_shape=(None, 32, 32, 128))
    model.summary()
