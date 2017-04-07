from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, BatchNormalization, merge, ELU, LeakyReLU, UpSampling2D, Conv2D, Concatenate, add, Lambda
import keras.backend as K
import tensorflow as tf


def residual_block(input_, channels, kxs=None, kys=None, id=0, cat=False, scope=''):
    n_conv = len(channels)
    if kxs is None:
        kxs = [3] * n_conv
    if kys is None:
        kys = [3] * n_conv
    x = input_
    x = convolution_blocks(x, channels, kxs, kys, id,)
    if cat:
        m = merge([x, input_], mode='concat', name="resi_%d_merge" % id)
    else:
        m = merge([x, input_], mode='sum', name="resi_%d_merge" % id)
    return m


def upscale_block(input_, rx, ry, channel, id=0):
    x = input_
    x = Convolution2D(channel, 5, 5, border_mode='same',
                      name='upscale_%d_conv_0' % id)(x)
    x = ELU(name='upscale_%d_elu_0' % id)(x)
    x = UpSampling2D(size=(rx, ry), name='upscale_%d_upsample' % id)(x)
    # x = SubPixelUpscaling(r=2, channels=32)(x)
    x = Convolution2D(channel, 5, 5, border_mode='same',
                      name='upscale_%d_conv_1' % id)(x)
    x = ELU(name='upscale_%d_elu_1' % id)(x)
    return x


# def convolution_blocks(ip, nb_filters, nb_rows=None, nb_cols=None, subsamples=None, id=0, border_mode='same', scope=''):
#     """ stack of standard convolution blocks """
#     nb_layers = len(nb_filters)
#     x = ip
#     if subsamples is None:
#         subsamples = [(1, 1)] * nb_layers
#     if nb_rows is None:
#         nb_rows = [3]*nb_layers
#     if nb_cols is None:
#         nb_cols = [3]*nb_layers
#     for i in range(nb_layers):
#         x = convolution_block(ip=x,
#                               nb_filter=nb_filters[i],
#                               nb_row=nb_rows[i],
#                               nb_col=nb_cols[i],
#                               subsamples=subsamples,
#                               id=i,
#                               scope=scope + "convolutions%d/" % id)
#     return x

def conv_blocks(input_shape, nb_filters, kernel_size, id=0, padding='same'):
    nb_layers = len(nb_filters)
    if nb_layers < 2:
        raise ValueError(
            "conv_blocks requires nb_layers >= 2, while input is {0}.".format(nb_filters))
    m = Sequential()
    m.add(Conv2D(nb_filters[0], kernel_size=kernel_size,
                 padding=padding,  input_shape=input_shape, activation='elu'))
    m.add(BatchNormalization())
    for i in range(nb_layers - 2):
        m.add(Conv2D(
            nb_filters[i + 1], kernel_size=kernel_size, padding=padding, activation='elu'))
        m.add(BatchNormalization())
    m.add(Conv2D(nb_filters[-1], kernel_size=kernel_size, padding=padding))
    return m


def CELU():
    def celu_kernel(x):
        pos = K.elu(x)
        neg = K.elu(-x)
        return K.concatenate([pos, neg], axis=-1)

    def celu_output_shape(input_shape):
        shape = list(input_shape)
        if len(shape) != 4:
            raise TypeError(
                'input_shape must be 4d, got {0}.'.format(input_shape))
        shape[3] *= 2
        return tuple(shape)

    return Lambda(celu_kernel, output_shape=celu_output_shape)

def conv_f(x, filters, kernel_size, is_active=True, name='conv_f'):
    with tf.name_scope(name):
        x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
        if is_active:
            x = CELU()(x)
            x = BatchNormalization()(x)
    return x

def conv_block(x, filters, name='conv_block'):
    with tf.name_scope(name):
        t0 = x
        rep1 = x
        for i in range(5):
            rep1 = conv_f(rep1, filters*4, 1, True, 'conv_1_1_%d'%i)
        rep3 = x
        for i in range(3):
            rep3 = conv_f(rep3, filters*2, 3, True, 'conv_1_3_%d'%i)
        rep5 = conv_f(rep3, filters, 5, True, 'conv_1_5_0')
        x = Concatenate(name='cat1')([x, rep1, rep3, rep5])
        x = conv_f(filters, filters, 1, False, name='conv_cat_1')
        t1 = add([t0, x])
        x = t1
        rep1 = x
        for i in range(5):
            rep1 = conv_f(rep1, filters*4, 1, True, 'conv_2_1_%d'%i)
        rep3 = x
        for i in range(3):
            rep3 = conv_f(rep3, filters*2, 3, True, 'conv_2_3_%d'%i)
        rep5 = conv_f(rep3, filters, 5, True, 'conv_2_5_0')
        x = Concatenate(name='cat2')([x, rep1, rep3, rep5])
        x = conv_f(filters, filters, 1, False, name='conv_cat_2')
        t2 = add([t1, x])
        x = t2
        rep1 = x
        for i in range(5):
            rep1 = conv_f(rep1, filters*4, 1, True, 'conv_3_1_%d'%i)
        rep3 = x
        for i in range(3):
            rep3 = conv_f(rep3, filters*2, 3, True, 'conv_3_3_%d'%i)
        rep5 = conv_f(rep3, filters, 5, True, 'conv_3_5_0')
        x = Concatenate(name='cat3')([x, rep1, rep3, rep5])
        x = conv_f(filters, filters, 1, False, name='conv_cat_3')
        t3 = add([t2, x])
        x = Concatenate()([t0, t1, t2, t3])
    return x

def sr_base(x, down0, down1, name='sr2x'):
    with tf.name_scope(name):
        ups = UpSampling2D(size=(down0, down1))(x)
        rep = conv_block(x, 64)
        rep_ups = UpSampling2D(size=(down0, down1))(rep)
        res_inf = conv_f(rep_ups, 1, 5, False, name='res_inf')
        inf = add([ups, res_inf])
    return inf

def convolution_block(ip, nb_filter, nb_row, nb_col, subsample=(1, 1), id=0, border_mode='same', scope=''):
    """ standard convolution block """
    x = Convolution2D(nb_filter,  nb_row, nb_col, border_mode=border_mode,
                      name=scope + 'conv%d/conv' % id, subsample=subsample)(ip)
    x = BatchNormalization(name=scope + 'conv%d/bn' % id)(x)
    x = ELU(name=scope + "conv%d/elu" % id)(x)
    return x
