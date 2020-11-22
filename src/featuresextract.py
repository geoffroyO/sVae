# TensorFlow and tf.keras
from tensorflow.keras import layers
from tensorflow.python.keras import Input
import six
import functools
from tensorflow.python.keras.backend import variable, concatenate, l2_normalize
from tensorflow.python.keras.layers import Conv2D, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.base import InputSpec
from tensorflow.python.ops import nn_ops

import numpy as np


class CombinedConv2D(Conv2D):
    def __init__(self, filters,
                 kernel_size=(5, 5),
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None, **kwargs):

        super().__init__(
            filters,
            kernel_size,
            strides,
            padding,
            data_format,
            dilation_rate,
            groups,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.regular_kernel = None
        self.srm_kernel = None
        self.bias = None
        self.kernel = None
        self._convolution_op = None
        self.bayar_kernel = None

    def _get_srm_list(self):

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

    def _build_SRM_kernel(self):
        kernel = []
        srm_list = self._get_srm_list()
        for srm in srm_list:
            for ch in range(3):
                this_ch_kernel = np.zeros([5, 5, 3]).astype('float32')
                this_ch_kernel[:, :, ch] = srm
                kernel.append(this_ch_kernel)
        kernel = np.stack(kernel, axis=-1)
        srm_kernel = variable(kernel, dtype='float32', name='srm')
        return srm_kernel

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        filters = self.filters - 15 - 3
        if filters >= 1:
            regular_kernel_shape = self.kernel_size + (input_dim, filters)
            self.regular_kernel = self.add_weight(
                name='regular_kernel',
                shape=regular_kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True)
        else:
            self.regular_kernel = None

        self.srm_kernel = self._build_SRM_kernel()
        bayar_kernel_shape = self.kernel_size + (input_dim, 3)
        self.bayar_kernel = self.add_weight(shape=bayar_kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='bayar_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=BayarConstraint(),
                                            trainable=False)

        if self.regular_kernel is not None:
            all_kernels = [self.regular_kernel, self.srm_kernel, self.bayar_kernel]
        else:
            all_kernels = [self.srm_kernel, self.bayar_kernel]
        self.kernel = concatenate(all_kernels, axis=-1)
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)
        tf_op_name = self.__class__.__name__
        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

def create_featex_vgg16_base(type=1):
    base = 32
    img_input = Input(shape=(32, 32, 3), name='image_in')

    # block 1
    bname = 'b1'
    nb_filters = base
    x = CombinedConv2D(nb_filters if type in [0,1] else 16, activation='relu', use_bias=False, padding='same',
                       name=bname+'c1')(img_input)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)

    # block 2
    bname = 'b2'
    nb_filters = 2 * base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c1')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)

    # block 3
    bname = 'b3'
    nb_filters = 4 * base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c1')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c3')(x)

    # block 4
    bname = 'b4'
    nb_filters = 8 * base
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c1')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c3')(x)

    # block 5
    bname = 'b5'
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c1')(x)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=bname + 'c2')(x)
    activation = None if type >= 1 else 'tanh'
    print("INFO: use activation in the last CONV={}".format(activation))
    sf = Conv2D(nb_filters, (3, 3), activation=activation, padding='same', name='transform')(x)
    sf = Lambda(lambda t: l2_normalize(t, axis=-1), name='L2')(sf)
    return Model(img_input, sf, name='Featex')
