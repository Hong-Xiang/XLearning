from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Convolution2D, BatchNormalization, merge, ELU, LeakyReLU, UpSampling2D


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


def convolution_blocks(ip, nb_filters, nb_rows, nb_cols, id=0, border_mode='same', scope=''):
    """ stack of standard convolution blocks """
    nb_layers = len(nb_filters)
    x = ip
    for i in range(nb_layers):
        x = convolution_block(x, nb_filters[i], nb_rows[i], nb_cols[
                              i], i, scope=scope+"convolutions%d/"%id)
    return x


def convolution_block(ip, nb_filter, nb_row, nb_col, id=0, border_mode='same', scope=''):
    """ standard convolution block """
    x = Convolution2D(nb_filter,  nb_row, nb_col, border_mode=border_mode,
                      name=scope + 'conv%d/conv' % id)(ip)
    x = BatchNormalization(name=scope + 'conv%d/bn' % id)(x)
    x = ELU(name=scope + "conv%d/elu" % id)(x)
    return x
