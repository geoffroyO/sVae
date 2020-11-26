import anodec
import lightfeaturesextract as lf

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.losses.losses_impl import Reduction, absolute_difference


def vaeLoss(features, z_mu, z_sigma, reconstruction):
    kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.math.square(z_mu) +
                                                 tf.math.square(z_sigma) - tf.math.log(tf.square(z_sigma)) - 1,
                                                 axis=1))
    L1 = absolute_difference(features, reconstruction, reduction=Reduction.NONE)
    reconstrution_loss = tf.reduce_mean(tf.reduce_sum(L1, axis=[1, 2, 3]))
    return reconstrution_loss, kl_loss


class falseModel(Model):
    def __init__(self, featex, encoder, decoder):
        super(falseModel, self).__init__()
        self.featex = featex
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, svae_optimizer, vaeLoss):
        super(falseModel, self).compile()
        self.vae_optimizer = svae_optimizer
        self.vaeLoss = vaeLoss


    def call(self, inputs):
        features = self.featex(inputs)
        z_mu, z_log_sigma, z_sigma, z = self.encoder(features)
        output = self.decoder(z)
        return output

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        features = self.featex(data)

        with tf.GradientTape() as tape:
            z_mu, z_log_sigma, z_sigma, z = self.encoder(features)
            reconstruction = self.decoder(z)
            reconstrution_loss, kl_loss = self.vaeLoss(features, z_mu, z_sigma, reconstruction)
            vae_loss = reconstrution_loss + kl_loss
        grads = tape.gradient(vae_loss, self.trainable_weights)
        self.vae_optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )
        return {
            "total_loss": vae_loss,
            "reconstruction_loss": reconstrution_loss,
            "kl_loss": kl_loss,
        }



def load_dsvae(dir):
    featex = lf.load_featex(dir)
    encoder = anodec.encoder()
    decoder = anodec.decoder()
    model = falseModel(featex, encoder, decoder)
    return model
