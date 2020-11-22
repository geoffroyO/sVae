# TensorFlow and tf.keras

from tensorflow.python.keras import Input

from tensorflow.python.keras.backend import variable
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.python.keras.models import Model

import numpy as np


def _get_srm_list():
    hsb = np.zeros([5, 5]).astype('float32')
    hsb[2, 1:] = np.array([[1, -3, 3, -1]]).astype('float32')
    hsb /= 3

    vbh = np.zeros([5, 5]).astype('float32')
    vbh[:4, 2] = np.array([1, -3, 3, 1]).astype('float32')
    vbh /= 3

    rf0 = np.zeros([5, 5]).astype('float32')
    rf0[1:4, 1:4] = np.array([[-1, 3, -1],
                              [2, -4, 2],
                              [-1, 2, -1]]).astype('float32')
    rf0 /= 4

    rf1 = np.array([[-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]]).astype('float32')
    rf1 /= 12

    rf2 = np.zeros([5, 5]).astype('float32')
    rf2[2, 1:4] = np.array([1, -2, 1])
    rf2 /= 2
    return [hsb, vbh, rf0, rf1, rf2]


def _build_SRM_kernel(shape, dtype=None):
    kernel = []
    srm_list = _get_srm_list()
    for srm in srm_list:
        for ch in range(3):
            this_ch_kernel = np.zeros([5, 5, 3]).astype('float32')
            this_ch_kernel[:, :, ch] = srm
            kernel.append(this_ch_kernel)
    kernel = np.stack(kernel, axis=-1)
    assert kernel.shape == shape
    srm_kernel = variable(kernel, dtype='float32', name='srm')

    return srm_kernel


def light_featex():
    base = 32
    img_input = Input(shape=(32, 32, 3), name='image_in')

    # block 1
    bname = 'b1'
    nb_filters = base

    conv = Conv2D(filters=17, kernel_size=5, padding='same', name='c1_bis')(img_input)
    x = Conv2D(filters=15,
               kernel_size = 5,
               kernel_initializer=_build_SRM_kernel,
               padding='same',
               name=bname+'c1',
               trainable=False)(img_input)
    x = Concatenate()([x, conv])
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname+'c2')(x)
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

    # block 1
    bname = 'b1'
    nb_filters = base

    conv = Conv2D(filters=17, kernel_size=5, padding='same', name='c1_bis')(img_input)
    x = Conv2D(filters=15,
               kernel_size = 5,
               kernel_initializer=_build_SRM_kernel,
               padding='same',
               name=bname+'c1',
               trainable=False)(img_input)
    x = Concatenate()([x, conv])
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname+'c2')(x)
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

if __name__ == '__main__':
    model = light_featex()
    model.summary()
