import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope
# import tensorflow.contrib.framework.python.framework


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

def upsampling2d(inputs, size, method=None, filters=None, name='upsampling'):
    if method is None:
        method = 'bilinear'
    input_shape = inputs.shape.as_list()
    h0, w0 = input_shape[1:3]
    h1 = int(h0 * size[0])
    w1 = int(w0 * size[1])
    with tf.name_scope(name):
        if method == 'nearest':
            h = tf.image.resize_nearest_neighbor(inputs, size=[h1, w1])
        elif method == 'bilinear':
            h = tf.image.resize_bilinear(inputs, size=[h1, w1])
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


# def conv2d(inputs,
#            filters,
#            kernel_size,
#            strides=(1, 1),
#            activation=tf.nn.elu,
#            padding='same',
#            data_format='channels_last',
#            is_bn=True,
#            training=True,
#            name='conv2d',
#            **kwargs):
#     with tf.name_scope(name) as scope:
#         h = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
#                              strides=strides, padding=padding, data_format=data_format, name=scope[:-1])
#         if is_bn:
#             if data_format == 'channels_first':
#                 axis = 1
#             else:
#                 axis = -1
#             h = tf.layers.batch_normalization(
#                 h, axis=axis, training=training, name=scope + 'bn')
#         with tf.name_scope('activation'):
#             h = activation(h)
#     return h


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


def residual2(inputs, filters=64, activation=tf.nn.elu, scale=0.1, name='res_u', reuse=None):
    with tf.name_scope('residual_unit'):
        with tf.variable_scope(name, reuse=reuse):
            h = tf.layers.conv2d(
                inputs, filters, 3, padding='same', activation=activation, name='conv0')
            h = tf.layers.conv2d(h, filters, 1, padding='same',
                                 activation=activation, name='conv1')
            h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv2')
        with tf.name_scope('shorcut'):
            out = scale * h + inputs
    return out


def celu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    with tf.name_scope('celu'):
        x = tf.nn.elu(tf.concat([x, -x], axis=-1))
    return x


@add_arg_scope
def conv2d(inputs, filters, **kwargs):
    defaults = {
        'padding': 'same',
        'kernel_size': 3
    }
    defaults.update(kwargs)
    return tf.layers.conv2d(inputs, filters, **defaults)


@add_arg_scope
def bn(*args, **kwargs):
    return tf.layers.batch_normalization(*args, **kwargs)


@add_arg_scope
def res_inc_unit(inputs, activation=celu, normalization=None, training=True, reuse=None, scale=1.0, name=None):
    filters = inputs.shape.as_list()[-1]
    fc = filters // 4
    default_name = 'residual_inception_unit'
    with tf.variable_scope(name, default_name, reuse=reuse):
        with arg_scope([conv2d], padding='same'):
            with arg_scope([bn], training=training):
                # with tf.name_scope(name, default_name):
                # path 0
                with tf.variable_scope('path_0'):
                    h0 = inputs
                    if normalization == 'bn':
                        h0 = bn(h0, name='bn')
                    h0 = activation(h0)
                    h0 = conv2d(h0, fc, 1, name='conv')

                # path 1
                with tf.variable_scope('path_1'):
                    h1 = inputs
                    kernel_sizes = [1, 3]
                    for i, ks in enumerate(kernel_sizes):
                        if normalization == 'bn':
                            h1 = bn(h1, name='bn_%d' % i)
                        h1 = activation(h1)
                        h1 = conv2d(h1, fc, ks, name='conv_%d' % i)

                # path 2
                with tf.variable_scope('path_2'):
                    h2 = inputs
                    kernel_sizes = [1, 3, 3]
                    for i, ks in enumerate(kernel_sizes):
                        if normalization == 'bn':
                            h2 = bn(h2, name='bn_%d' % i)
                        h2 = activation(h2)
                        h2 = conv2d(h2, fc, ks, name='conv_%d' % i)

                # concate
                with tf.name_scope('concate'):
                    h = tf.concat([h0, h1, h2], axis=-1)
                with tf.variable_scope('path_merge'):
                    kernel_sizes = [1, 3]
                    for i, ks in enumerate(kernel_sizes):
                        if normalization == 'bn':
                            h = bn(h, name='bn_%d' % i)
                        h = activation(h)
                        h = conv2d(h, filters, ks, name='conv_%d' % i)
                with tf.name_scope('add'):
                    out = h * scale + inputs
    return out


@add_arg_scope
def res2_unit(inputs, activation=tf.nn.elu, normalization=None, training=True, scale=0.1, reuse=None, name=None):
    default_name = 'residual2_unit'
    filters = inputs.shape.as_list()[-1]
    with tf.name_scope(name, default_name):
        with tf.variable_scope(name, reuse=reuse):
            h = tf.layers.conv2d(
                inputs, filters, 3, padding='same', activation=activation, name='conv0')
            h = tf.layers.conv2d(h, filters, 1, padding='same',
                                 activation=activation, name='conv1')
            h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv2')
        with tf.name_scope('shorcut'):
            out = scale * h + inputs
    return out


@add_arg_scope
def stack_res_units(inputs, depths, basic_unit=res_inc_unit, activation=celu, normalization=None, training=None, scale=1.0, reuse=None, name=None):
    default_name = 'stack_res_units'
    with tf.variable_scope(name, default_name, reuse=reuse):
        h = inputs
        for i in range(depths):
            h = basic_unit(h, activation=activation, normalization=normalization, training=training, scale=scale, name='unit_%d'%i)
    return h

@add_arg_scope
def stem(inputs, filters, reuse=None, name=None):
    with tf.variable_scope(name, 'stem', reuse=reuse):
        out = conv2d(inputs, filters)
    return out

@add_arg_scope
def incept(inputs, filters, nb_path=3, activation=None, normalization=None, training=None, reuse=None, name=None):
    default_name = 'incept'
    fp = filters // nb_path
    with tf.variable_scope(name, default_name, reuse=reuse):
        with arg_scope([conv2d], padding='same'):
            with arg_scope([bn], training=training):
                # with tf.name_scope(name, default_name):
                h_paths = []
                for ipath in range(nb_path):
                    with tf.variable_scope('path_%d'%ipath):
                        h = inputs
                        kernel_sizes = [1] + [3]*ipath
                        for i, ks in enumerate(kernel_sizes):
                            if normalization == 'bn':
                                h = bn(h, name='bn_%d' % i)
                            h = activation(h)
                            h = conv2d(h, fp, kernel_size=ks, name='conv_%d' % i)
                        h_paths.append(h)
                # concate
                with tf.name_scope('concate'):
                    h = tf.concat(h_paths, axis=-1)
                with tf.variable_scope('path_merge'):
                    kernel_sizes = [1]
                    for i, ks in enumerate(kernel_sizes):
                        if normalization == 'bn':
                            h = bn(h, name='bn_%d' % i)
                        h = activation(h)
                        h = conv2d(h, filters, kernel_size=ks, name='conv_%d' % i)
    return h

@add_arg_scope
def residual(inputs, basic_unit, basic_unit_args, scale=None, reuse=None, name=None):
    default_name = 'residual'
    filters = inputs.shape.as_list()[-1]
    basic_unit_args = dict(basic_unit_args)
    basic_unit_args.update({'filters': filters})
    if scale is None:
        scale = (1.0, 1.0)
    with tf.variable_scope(name, default_name, reuse=reuse):        
        h = basic_unit(inputs, **basic_unit_args)
        with tf.name_scope('add'):
            out = scale[0]*inputs + scale[1]*h
    return out

@add_arg_scope
def stack_residuals(inputs, depths, residual_args, reuse=None, name=None):
    default_name = 'stack_residual'
    with tf.variable_scope(name, default_name, reuse=reuse):
        h = inputs
        for i in range(depths):
            # name = residual_args.get('name')
            # kwargs = dict(residual_args)
            # kwargs.pop('name')
            # if name is None:
            #     name = 'residual_%d'%i
            # else:
            #     name = name[i]            
            h = residual(h, name='residual_%d'%i, **residual_args)
        out = h
    return out

@add_arg_scope
def channel_change(inputs, filters, normalization=None, training=None, reuse=None, name=None):
    default_name = 'channel_change'
    fc = filters // 4
    with tf.variable_scope(name, default_name, reuse=reuse):
        with arg_scope([conv2d], padding='same'):
            with arg_scope([bn], training=training):
                # with tf.name_scope(name, default_name):
                # path 0
                with tf.variable_scope('path_0'):
                    h0 = inputs
                    if normalization == 'bn':
                        h0 = bn(h0, name='bn')
                    h0 = activation(h0)
                    h0 = conv2d(h0, fc, 1, name='conv')

                # path 1
                with tf.variable_scope('path_1'):
                    h1 = inputs
                    kernel_sizes = [1, 3]
                    for i, ks in enumerate(kernel_sizes):
                        if normalization == 'bn':
                            h1 = bn(h1, name='bn_%d' % i)
                        h1 = activation(h1)
                        h1 = conv2d(h1, fc, ks, name='conv_%d' % i)

                # path 2
                with tf.variable_scope('path_2'):
                    h2 = inputs
                    kernel_sizes = [1, 3, 3]
                    for i, ks in enumerate(kernel_sizes):
                        if normalization == 'bn':
                            h2 = bn(h2, name='bn_%d' % i)
                        h2 = activation(h2)
                        h2 = conv2d(h2, fc, ks, name='conv_%d' % i)

                # concate
                with tf.name_scope('concate'):
                    h = tf.concat([h0, h1, h2], axis=-1)
                with tf.variable_scope('path_merge'):
                    kernel_sizes = [1, 3]
                    for i, ks in enumerate(kernel_sizes):
                        if normalization == 'bn':
                            h = bn(h, name='bn_%d' % i)
                        h = activation(h)
                        h = conv2d(h, filters, ks, name='conv_%d' % i)
    return out

@add_arg_scope
def sr_infer(low_res, reps, size, reuse=None, name=None):
    with tf.variable_scope(name, 'sr_infer', reuse=reuse):
        interp = upsampling2d(low_res, size)
        residual = conv2d(reps, 1, kernel_size=1, name='residual')
        inference = interp + residual
    return inference, residual, interp

@add_arg_scope
def super_resolution_unit(low_resolution,
                          filters,
                          depths,
                          reps=None,
                          normalization=None,
                          training=True,
                          activation=celu,
                          down_sample_ratio=(2, 2),
                          upsample_method=None,
                          basic_unit=res_inc_unit,
                          reuse=None,
                          scale=0.1,
                          name=None):
    """ super resution block construsted by stack of basic units:
        Supported basic units:
            res2_unit
            res_inc_unit
        Return: A dict of tensors:
            inference:
            residual:
            interpolation:
            representations:
    """
    default_name = 'super_resolution_unit'
    basic_unit_args = {
        'filters': filters,
        'activation': activation
    }

    if basic_unit == res2_unit:
        basic_unit_args.update({'scale': scale})
    else:
        basic_unit_args.update({'normalization': normalization,
                                'training': training})
    with tf.variable_scope(name, default_name):
        with tf.variable_scope('input'):
            if reps is None:
                h = low_resolution
                with tf.variable_scope('stem'):
                    h = conv2d(
                        h, basic_unit_args['filters'], kernel_size=5)
            else:
                h = reps
        for id_depth in range(depths):
            with tf.variable_scope(None, basic_unit.__name__):
                h = basic_unit(h, **basic_unit_args)
        with tf.variable_scope('upsample_reps'):
            h = upsampling2d(h, size=down_sample_ratio, method=upsample_method)
        with tf.variable_scope('interpolation'):
            interp = upsampling2d(
                low_resolution, down_sample_ratio, method='nearest')
        with tf.variable_scope(None, basic_unit.__name__):
            reps_out = basic_unit(h, **basic_unit_args)
        with tf.variable_scope('inference'):
            with tf.variable_scope('conv_residual'):
                residual = conv2d(reps_out, 1, 1, padding='same')
            with tf.variable_scope('add'):
                inference = residual + interp
    out = {'inference': inference, 'residual': residual,
           'interpolation': interp, 'representations': reps_out}
    return out

@add_arg_scope
def crop(inputs, crop_size=None, name='crop'):
    with tf.name_scope(name):
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        target_shape = inputs.shape.as_list()
        target_shape[1] -= 2 * crop_size[0]
        target_shape[2] -= 2 * crop_size[1]
        out = tf.slice(inputs, [0, crop_size[0], crop_size[1], 0], target_shape)        
    return out

@add_arg_scope
def crop_many(inputs, crop_size=None, batch_size=None, name='crop_many'):
    """Crop tensors.
        If crop_size is None, tensors will be center cropped to same size as inputs[0].
    """
    if crop_size is None:
        keep_0 = True
        if len(inputs) < 2:
            raise ValueError(
                'Crop size must be specified when nb of inputs < 2.')
        else:
            crop_size = []
            sp0 = inputs[0].shape.as_list()
            sp1 = inputs[1].shape.as_list()
            csx = (sp1[1] - sp0[1]) // 2
            csy = (sp1[2] - sp0[2]) // 2
            crop_size = [csx, csy]
        target_shape = inputs[0].shape.as_list()
        if target_shape[0] is None:
            target_shape[0] = batch_size
    else:
        keep_0 = False
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        target_shape = inputs[0].shape.as_list()
        target_shape[1] -= 2 * crop_size[0]
        target_shape[2] -= 2 * crop_size[1]

    out = []    
    for i, t in enumerate(inputs):
        if keep_0 and i == 0:
            out.append(t)
            continue
        t_c = tf.slice(t, [0, crop_size[0], crop_size[1], 0], target_shape)
        out.append(t_c)
    return out

@add_arg_scope
def upsampling2d_many(inputs, size, method='nearest', reuse=None, name=None):
    out = []
    with tf.variable_scope(name, 'upsampling2d_many', reuse=reuse):
        for t in inputs:
            tu = upsampling2d(t, size=size, method=method)
            out.append(tu)
    return out


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))
