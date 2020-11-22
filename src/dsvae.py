import anodec
import lightfeaturesextract as lf

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import log
from tensorflow.python.keras.losses import mse
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Subtract
from tensorflow.python.ops.losses.losses_impl import Reduction, absolute_difference


def predConv2D():
    input = Input(shape=(32, 32, 128))
    pred_Conv2D = Conv2D(1, (7, 7), padding='same', activation='sigmoid', name='pred')(input)
    return Model(input, pred_Conv2D, name='pred-Conv')


def predLoss(labels, predictions):
    return tf.reduce_mean(mse(labels, predictions))


def vaeLoss(features, z_mean, z_var, reconstruction):
    kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_var) - log(tf.square(z_var)) - 1, axis=1))
    L1 = absolute_difference(features, reconstruction, reduction=Reduction.NONE)
    reconstrution_loss = tf.reduce_mean(tf.reduce_sum(L1, axis=[1, 2, 3]))
    return reconstrution_loss, kl_loss


class falseModel(Model):
    def __init__(self, featex, encoder, decoder):
        super(falseModel, self).__init__()
        self.featex = featex
        self.encoder = encoder
        self.decoder = decoder
        self.subtract = Subtract()
        self.predConv = predConv2D()

    def compile(self, svae_optimizer, pred_optimizer, predLoss, vaeLoss):
        super(falseModel, self).compile()
        self.svae_optimizer = svae_optimizer
        self.pred_optimizer = pred_optimizer
        self.predLoss = predLoss
        self.vaeLoss = vaeLoss

    def call(self, inputs):
        features = self.featex(inputs)
        x = self.encoder(features)
        x = self.decoder(x)
        x = self.subtract([x, features])
        output = self.predConv(x)
        return output

    def train_step(self, data):
        if isinstance(data, tuple):
            labels = data[1]
            data = data[0]

        features = self.featex(data)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z_var, z = self.encoder(features)
            reconstruction = self.decoder(z)
            reconstrution_loss, kl_loss = self.vaeLoss(features, z_mean, z_var, reconstruction)
            vae_loss = reconstrution_loss + kl_loss
        grads = tape.gradient(vae_loss, self.encoder.trainable_weights)
        self.svae_optimizer.apply_gradients(
            zip(grads, self.encoder.trainable_weights)
        )

        with tf.GradientTape() as tape:
            sub = self.subtract([features, reconstruction])
            output = self.predConv(sub)
            pred_loss = self.predLoss(labels, output[:, :, :, 0])
        grads = tape.gradient(pred_loss, self.predConv.trainable_weights)
        self.pred_optimizer.apply_gradients(
            zip(grads, self.predConv.trainable_weights)
        )

        return {
            "total_loss": vae_loss + pred_loss,
            "reconstruction_loss": reconstrution_loss,
            "kl_loss": kl_loss,
            "pred_loss": pred_loss
        }


def load_dsvae(dir):
    featex = lf.load_featex(dir)
    encoder = anodec.encoder()
    decoder = anodec.decoder()
    model = falseModel(featex, encoder, decoder)
    return model


if __name__ == '__main__':
    print("ok")
