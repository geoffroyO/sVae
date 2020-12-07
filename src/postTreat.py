import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import BatchNormalization, Subtract, Reshape


class postTreat(keras.Model):
    def __init__(self, anodec, **kwargs):
        super(postTreat, self).__init__(**kwargs)
        self.anodec = anodec
        self.batchNorm = BatchNormalization()
        self.subtract = Subtract()
        self.reshape = Reshape((32, 32, 1))

    def call(self, input):
        features = self.anodec.featex(input)
        anoFeat = self.anodec(input)

        squareFeat = tf.math.square(features)
        squareanoFeat = tf.math.square(anoFeat)

        sumFeat = tf.reduce_sum(squareFeat, axis=-1)
        sumanoFeat = tf.reduce_sum(squareanoFeat, axis=-1)

        sqrtFeat = tf.math.sqrt(sumFeat)
        sqrtanoFeat = tf.math.sqrt(sumanoFeat)

        sub = self.subtract([sqrtFeat, sqrtanoFeat])

        return sub


