# TensorFlow and tf.keras
import cv2
from tensorflow.python.keras import Input

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, Subtract
from tensorflow.python.keras.models import Model

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


def _gaussian_kernel(size, mean, std,):
    d = tfp.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    gauss_kernel = np.asarray(gauss_kernel, dtype=float)
    gauss_kernel = [[gauss_kernel, gauss_kernel, gauss_kernel],
                    [gauss_kernel, gauss_kernel, gauss_kernel],
                    [gauss_kernel, gauss_kernel, gauss_kernel]]
    gauss_kernel = np.einsum('klij->ijlk', gauss_kernel)
    gauss_kernel = gauss_kernel.flatten()
    initializer_gauss = tf.constant_initializer(gauss_kernel)

    return initializer_gauss


def light_featex():
    base = 32
    img_input = Input(shape=(32, 32, 3), name='image_in')

    blur = Conv2D(filters=3,
               kernel_size=[5, 5],
               kernel_initializer=_gaussian_kernel(2, 0, 11),
               padding='same',
               name='gaussian_blur',
               trainable=False)(img_input)

    blur = Conv2D(filters=3,
               kernel_size=[5,5],
               kernel_initializer=_build_SRM_kernel(),
               padding='same',
               name='srm_blur',
               trainable=False)(blur)

    x = Conv2D(filters=3,
               kernel_size=[5,5],
               kernel_initializer=_build_SRM_kernel(),
               padding='same',
               name='srm',
               trainable=False)(img_input)

    x = Subtract()([x, blur])
    # block 1
    bname = 'b1'
    nb_filters = base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname+'c1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # block 2
    bname = 'b2'
    nb_filters = 2 * base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname+'c1')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname+'c2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # block 3
    bname = 'b3'
    nb_filters = 4 * base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname+'c1')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname+'c2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Flatten(name='classifier_flatten')(x)
    x = Dense(1024, activation='relu', name='classifier_densee')(x)
    x = Dropout(0.5)(x)
    sf = Dense(1, activation='sigmoid')(x)

    return Model(img_input, sf, name='Featex')


def featex():
    base = 32
    img_input = Input(shape=(32, 32, 3), name='image_in')

    blur = Conv2D(filters=3,
                  kernel_size=[5, 5],
                  kernel_initializer=_gaussian_kernel(2, 0, 11),
                  padding='same',
                  name='gaussian_blur',
                  trainable=False)(img_input)

    blur = Conv2D(filters=3,
                  kernel_size=[5, 5],
                  kernel_initializer=_build_SRM_kernel(),
                  padding='same',
                  name='srm_blur',
                  trainable=False)(blur)

    x = Conv2D(filters=3,
               kernel_size=[5, 5],
               kernel_initializer=_build_SRM_kernel(),
               padding='same',
               name='srm',
               trainable=False)(img_input)

    x = Subtract()([x, blur])
    # block 1
    bname = 'b1'
    nb_filters = base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # block 2
    bname = 'b2'
    nb_filters = 2 * base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c1')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # block 3
    bname = 'b3'
    nb_filters = 4 * base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c1')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    return Model(img_input, x, name='Featex')


def load_featex(dir):
    model_class = light_featex()
    model_class.load_weights(dir)

    model = featex()
    for index in range(len(model.layers)):
        model.layers[index].set_weights(model_class.layers[index].get_weights())
        model.layers[index].trainable = False
        if 'classifier' in model_class.layers[index].name:
            break
    return model


def load_all_featex(dir):
    model_class = light_featex()
    model_class.load_weights(dir)

    return model_class


if __name__ == '__main__':
    model = light_featex()
    model.summary()
