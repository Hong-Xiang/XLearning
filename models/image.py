from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, BatchNormalization, merge, ELU, LeakyReLU, UpSampling2D, Conv2D, Concatenate, add, Lambda, Cropping2D
import keras.backend as K
import tensorflow as tf

from ..models.merge import sub

# import tensorflow.contrib.framework.python.framework


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


# def upscale_block(input_, rx, ry, channel, id=0):
#     x = input_
#     x = Convolution2D(channel, 5, 5, border_mode='same',
#                       name='upscale_%d_conv_0' % id)(x)
#     x = ELU(name='upscale_%d_elu_0' % id)(x)
#     x = UpSampling2D(size=(rx, ry), name='upscale_%d_upsample' % id)(x)
#     # x = SubPixelUpscaling(r=2, channels=32)(x)
#     x = Convolution2D(channel, 5, 5, border_mode='same',
#                       name='upscale_%d_conv_1' % id)(x)
#     x = ELU(name='upscale_%d_elu_1' % id)(x)
#     return x

def upsampling2d(inputs, size, method='nearest', filters=None, name='upsampling'):
    input_shape = inputs.shape().as_list()
    h0, w0 = input_shape[1:3]
    h1 = int(h0 * size[0])
    w1 = int(w0 * size[1])
    with tf.name_scope(name):
        if method == 'nearest':
            h = tf.image.resize_nearest_neighbor(inputs, size=size)
        elif method == 'bilinear':
            h = tf.image.resize_bilinear(inputs, size=size)
        elif method == 'deconv':
            h = tf.layers.conv2d_transpose(
                inputs, filters, 3, strides=size, padding='same')
    return h

def enc_img(inputs, base_filters, data_format, reuse=None):
    conv_cfgs = {
        'activation': tf.nn.elu,
        'padding': 'same',
        'data_format': 'channel_first',
        'reuse': reuse
    }
    h = inputs
    f = base_filters
    with tf.name_scope('enc') as scope:
        h = tf.layers.conv2d(h, f, 3, **conv_cfgs, name=scope + 'conv1')
        h = tf.layers.conv2d(h, f, 3, **conv_cfgs, name=scope + 'conv2')
        h = tf.layers.conv2d(h, 2 * f, 2, 2, **conv_cfgs, name=scope + 'conv3')
        h = tf.layers.conv2d(h, 2 * f, 3, **conv_cfgs, name=scope + 'conv4')
        h = tf.layers.conv2d(h, 2 * f, 3, **conv_cfgs, name=scope + 'conv5')
        h = tf.layers.conv2d(h, 3 * f, 2, 2, **conv_cfgs, name=scope + 'conv6')
        h = tf.layers.conv2d(h, 3 * f, 3, **conv_cfgs, name=scope + 'conv7')
        h = tf.layers.conv2d(h, 3 * f, 3, **conv_cfgs, name=scope + 'conv8')
        h = tf.layers.conv2d(h, 4 * f, 2, 2, **conv_cfgs, name=scope + 'conv9')
        h = tf.layers.conv2d(h, 4 * f, 3, **conv_cfgs, name=scope + 'conv10')
        h = tf.layers.conv2d(h, 4 * f, 3, **conv_cfgs, name=scope + 'conv11')
        h = slim.flatten()
        shape_last = h.shape().to_list()
        dim_latent = int(np.prod(dim_latent[1:]))
        latent = tf.layers.dense(h, dim_latent, name=scope + 'dense')
    return latent


def dec_img(inputs, shape, base_filters, data_format, reuse=None):
    conv_cfgs = {
        'activation': tf.nn.elu,
        'padding': 'same',
        'data_format': data_format,
        'reuse': reuse
    }
    h = inputs
    f = base_filters
    with tf.name_scope('enc') as scope:
        h = tf.reshape(h, shape)
        h = tf.layers.conv2d(h, f, 3, **conv_cfgs, name=scope + 'conv1')
        h = tf.layers.conv2d(h, f, 3, **conv_cfgs, name=scope + 'conv2')
        h = tf.contrib.keras.layers.UpSampling2D(
            size=(2, 2), data_format=data_format)
        h = tf.layers.conv2d(h, 2 * f, 2, 2, **conv_cfgs, name=scope + 'conv3')
        h = tf.layers.conv2d(h, 2 * f, 3, **conv_cfgs, name=scope + 'conv4')
        h = tf.layers.conv2d(h, 2 * f, 3, **conv_cfgs, name=scope + 'conv5')
        h = tf.layers.conv2d(h, 3 * f, 2, 2, **conv_cfgs, name=scope + 'conv6')
        h = tf.layers.conv2d(h, 3 * f, 3, **conv_cfgs, name=scope + 'conv7')
        h = tf.layers.conv2d(h, 3 * f, 3, **conv_cfgs, name=scope + 'conv8')
        h = tf.layers.conv2d(h, 4 * f, 2, 2, **conv_cfgs, name=scope + 'conv9')
        h = tf.layers.conv2d(h, 4 * f, 3, **conv_cfgs, name=scope + 'conv10')
        h = tf.layers.conv2d(h, 4 * f, 3, **conv_cfgs, name=scope + 'conv11')
        h = slim.flatten()
        shape_last = h.shape().to_list()
        dim_latent = int(np.prod(dim_latent[1:]))
        latent = tf.layers.dense(h, dim_latent, name=scope + 'dense')
    return latent


def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           activation=tf.nn.elu,
           padding='same',
           data_format='channels_last',
           is_bn=True,
           training=True,
           name='conv2d',
           **kwargs):
    with tf.name_scope(name) as scope:
        h = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                             strides=strides, padding=padding, data_format=data_format, name=scope[:-1])
        if is_bn:
            if data_format == 'channels_first':
                axis = 1
            else:
                axis = -1
            h = tf.layers.batch_normalization(
                h, axis=axis, training=training, name=scope + 'bn')
        with tf.name_scope('activation'):
            h = activation(h)
    return h


def residual_unit(inputs, filters, kernel_size, strides=(1, 1), activation=tf.nn.elu, training=True, padding='same', data_format='channels_last', name='conv2d', nb_convs=2):
    h = inputs
    with tf.name_scope(name) as scope:
        for i in range(nb_convs):
            h = conv2d(h, filters, kernel_size, strides, activation,
                       training, padding, data_format, name='conv_%d' % i)


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


def conv_blocks(input_shape, nb_filters, kernel_size, id=0, padding='valid', is_celu=False, is_final_active=False, name='conv_blocks'):
    with tf.name_scope(name):
        if is_celu:
            activ = CELU
        else:
            activ = ELU
        nb_layers = len(nb_filters)
        if nb_layers < 2:
            raise ValueError(
                "conv_blocks requires nb_layers >= 2, while input is {0}.".format(nb_filters))
        m = Sequential(name=name)
        m.add(Conv2D(nb_filters[0], kernel_size=kernel_size,
                     padding=padding,  input_shape=input_shape))
        m.add(activ())
        # m.add(BatchNormalization())
        for i in range(nb_layers - 2):
            m.add(Conv2D(
                nb_filters[i + 1], kernel_size=kernel_size, padding=padding, activation='elu'))
            m.add(activ())
            # m.add(BatchNormalization())
        m.add(Conv2D(nb_filters[-1], kernel_size=kernel_size, padding=padding))
        if is_final_active:
            m.add(activ())
    return m


def sr_end(res, itp, ip_h, name='sr_end', is_res=True):
    """ Assuming shape(itp) == shape(ip_h)
    reps is center croped shape of itp/ip_h
    """
    with tf.name_scope(name):
        spo = res.shape.as_list()[1:3]
        spi = itp.shape.as_list()[1:3]
        cpx = (spi[0] - spo[0]) // 2
        cpy = (spi[1] - spo[1]) // 2
        crop_size = (cpx, cpy)
        itp_c = Cropping2D(crop_size)(itp)
        with tf.name_scope('output'):
            inf = add([res, itp_c])
        if is_res:
            with tf.name_scope('label_cropped'):
                ip_c = Cropping2D(crop_size)(ip_h)
            with tf.name_scope('res_out'):
                res_inf = sub(ip_c, inf)
            with tf.name_scope('res_itp'):
                res_itp = sub(ip_c, itp_c)
        else:
            res_inf = None
            res_itp = None
        return (inf, crop_size, res_inf, res_itp)


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
        x = Conv2D(filters, kernel_size=kernel_size, padding='valid')(x)
        if is_active:
            x = CELU()(x)
            x = BatchNormalization()(x)
    return x


def conv_block(x, filters, name='conv_block'):
    with tf.name_scope(name):
        t0 = x
        rep1 = x
        for i in range(5):
            rep1 = conv_f(rep1, filters * 4, 1, True, 'conv_1_1_%d' % i)
        rep3 = x
        for i in range(3):
            rep3 = conv_f(rep3, filters * 2, 3, True, 'conv_1_3_%d' % i)
        rep5 = conv_f(rep3, filters, 5, True, 'conv_1_5_0')
        x = Concatenate(name='cat1')([x, rep1, rep3, rep5])
        x = conv_f(filters, filters, 1, False, name='conv_cat_1')
        t1 = add([t0, x])
        x = t1
        rep1 = x
        for i in range(5):
            rep1 = conv_f(rep1, filters * 4, 1, True, 'conv_2_1_%d' % i)
        rep3 = x
        for i in range(3):
            rep3 = conv_f(rep3, filters * 2, 3, True, 'conv_2_3_%d' % i)
        rep5 = conv_f(rep3, filters, 5, True, 'conv_2_5_0')
        x = Concatenate(name='cat2')([x, rep1, rep3, rep5])
        x = conv_f(filters, filters, 1, False, name='conv_cat_2')
        t2 = add([t1, x])
        x = t2
        rep1 = x
        for i in range(5):
            rep1 = conv_f(rep1, filters * 4, 1, True, 'conv_3_1_%d' % i)
        rep3 = x
        for i in range(3):
            rep3 = conv_f(rep3, filters * 2, 3, True, 'conv_3_3_%d' % i)
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


def inception_residual_block(input_tensor, filters, is_final_activ=False, is_bn=True, activation=tf.nn.crelu, training=True, reuse=None, res_scale=0.1, name='irb'):
    cc = {'reuse': reuse, 'padding': 'same'}
    cb = {'reuse': reuse, 'training': training, 'scale': False}
    with tf.name_scope(name):
        with tf.name_scope('nin'):
            if activation == tf.nn.crelu:
                input_tensor = tf.layers.conv2d(
                    input_tensor, 2 * filters, 1, **cc)
            else:
                input_tensor = tf.layers.conv2d(input_tensor, filters, 1, **cc)
        h1 = tf.layers.conv2d(input_tensor, filters, 1, **cc)
        if is_bn:
            h1 = tf.layers.batch_normalization(h1, **cb)
        h1 = activation(h1)
        h1 = tf.layers.conv2d(h1, filters, 3, **cc)

        h2 = tf.layers.conv2d(input_tensor, filters, 1, **cc)
        if is_bn:
            h2 = tf.layers.batch_normalization(h2, **cb)
        h2 = activation(h2)
        h2 = tf.layers.conv2d(h2, filters, 3, **cc)
        if is_bn:
            h2 = tf.layers.batch_normalization(h2, **cb)
        h2 = activation(h2)
        h2 = tf.layers.conv2d(h2, filters, 3, **cc)

        h = tf.concat([h1, h2], axis=-1)
        if is_bn:
            h = tf.layers.batch_normalization(h, **cb)
        h = activation(h)
        h = tf.layers.conv2d(h, filters, 1, **cc)
        if is_bn:
            h = tf.layers.batch_normalization(h, **cb)
        h = activation(h)

        h = h * res_scale
        h = h + input_tensor
        if is_final_activ:
            h = activation(h)
        return h


def align_by_crop(target_tensor, input_tensors, batch_size=None, name='align_crop'):
    target_shape = target_tensor.shape.as_list()
    if batch_size is not None:
        target_shape[0] = batch_size
    ops = []
    with tf.name_scope(name):
        for t in input_tensors:
            input_shape = t.shape.as_list()
            crop_h = (input_shape[1] - target_shape[1]) // 2
            crop_w = (input_shape[2] - target_shape[2]) // 2
            target_shape_now = list(target_shape)
            target_shape_now[-1] = input_shape[-1]
            ops.append(tf.slice(t, [0, crop_h, crop_w, 0], target_shape_now))
    return ops
