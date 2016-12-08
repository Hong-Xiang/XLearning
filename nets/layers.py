from __future__ import absolute_import, division, print_function

from six.moves import xrange
import tensorflow as tf
import numpy as np
import xlearn.utils.general as utg

FLAGS = tf.app.flags.FLAGS


def _weight_variable(name, shape, ncolumn, scope=tf.get_variable_scope()):
    """Helper to create a Variable
    Args:
        name: name of the variable
        shape: list of ints
        ncolumn: number of outputs to be summarized.
        scope: variable scope
    Returns:
        tessor: a variable tensor
    """
    dtype = tf.float32
    # set optimal stdandard deviation for relu units.
    stddev = np.sqrt(2.0 / ncolumn)
    initer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    with tf.variable_scope(scope):
        var = tf.get_variable(
            name, shape=shape, initializer=initer, dtype=dtype)
    return var


def _weight_variable_with_decay(name, shape, ncolumn, wd=0.0, scope=tf.get_variable_scope()):
    """Helper to create a Variable with weight decay.    
    Args:
      name: name of the variable
      shape: list of ints
      ncolumn: number of outputs to be summarized.
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      tensor: a variable tensor
    """
    var = _weight_variable(name, shape, ncolumn, scope)
    if abs(wd) < FLAGS.eps:
        return var
    with tf.name_scope(name + 'weight_loss') as namescope:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd,
                              name=namescope + 'weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _moving_average_variable(name, shape, scope=None):
    if scope is None:
        scope = tf.get_variable_scope()
    dtype = tf.float32
    initer = tf.constant_initializer(0.0)
    with tf.variable_scope(scope):
        var = tf.get_variable(name, shape, initializer=initer,
                              dtype=dtype, trainable=False)
    ema = tf.train.ExponentialMovingAverage(decay=FLAGS.bn_decay)
    average_op = ema.apply([var])
    return ema, average_op


def _bias_variable(name, shape, scope=None, trainable=True):
    if scope is None:
        scope = tf.get_variable_scope()
    dtype = tf.float32
    initer = tf.constant_initializer(0.0)
    with tf.variable_scope(scope):
        var = tf.get_variable(name, shape, initializer=initer,
                              dtype=dtype, trainable=trainable)
    return var


def _placeholder(name, shape):
    dtype = tf.float32
    var = tf.placeholder(dtype, shape=shape, name=name)
    return var


def max_out(input0, input1, name=None):
    """Computes max out units: max(feature1, feature2)
    Args:
        - features1: A Tensor.
        _ features2: A Tensor with the same shape as features1.
        - name: A name for the operation (optional)
    Returns:
        A Tensor.
    """
    if name is None:
        name = "max_out"
    with tf.name_scope(name):
        output = tf.maximum(input0, input1, name=name)
    return output


def rrelu(tensor_input, name=None):
    """Computes random leaky rectified linear: max(features, r * features)
        - if FLAGS.is_train, r has a uniform distribution on [FLAGS.rrelu_min, FLAGS.rrelu_max]
        - else r = 1/2 (FLAGS.rrelu_min + FLAGS.rrelu_max)
    Args:
        - features: A Tensor.
        - name: A name for the operation (optional)
    Returns:
        A Tensor.
    """
    if name is None:
        name = 'RReLU'
    with tf.name_scope(name):
        if FLAGS.is_train:
            scalar = tf.random_uniform([],
                                       minval=FLAGS.rrelu_min,
                                       maxval=FLAGS.rrelu_max,
                                       dtype=tf.float32,
                                       name='random_ratio')
        else:
            scalar = (FLAGS.rrelu_min + FLAGS.rrelu_max) / 2
        leaked = tf.scalar_mul(scalar, tensor_input)
        tensor = tf.maximum(tensor_input, leaked, name='maximum')
    return tensor


def lrelu(features, name=None):
    """Computes leaky rectified linear: max(features, FLAGS.leak_ratio * features)
    Args:
        - features: A Tensor.
        - name: A name for the operation (optional)
    Returns:
        A Tensor.
    """
    if name is None:
        name = 'LReLU'
    with tf.name_scope(name):
        leaked = tf.scalar_mul(FLAGS.leak_ratio, features)
        tensor = tf.maximum(features, leaked, name='maximum')
    return tensor


def activation(tensor_input, activation_function=None, varscope=tf.get_variable_scope(), name='activation'):
    """Warp of activation functions"""
    if activation_function is None:
        if FLAGS.activation_function == "rrelu":
            activation_function = rrelu
        elif FLAGS.activation_function == "lrelu":
            activation_function = lrelu
        elif FLAGS.activation_function == "max_out":
            activation_function = max_out
        elif FLAGS.activation_function == "elu":
            activation_function = tf.nn.elu
        else:
            activation_function = tf.nn.relu
    if activation_function is tf.nn.relu:
        tensor = tf.nn.relu(tensor_input, name)
    if activation_function is lrelu:
        tensor = lrelu(tensor_input, name)
    return tensor


def matmul_bias(input_,
                shape,
                name="matmul_bias",
                varscope=tf.get_variable_scope()):
    """
    Basic full connect layer.

    Args:
        input_: input tensor
        shape: a list or tuple, [n_input, n_output]

    """
    with tf.name_scope(name) as scope:
        weights = _weight_variable_with_decay(scope + 'weights',
                                              shape=shape,
                                              ncolumn=shape[0],
                                              wd=FLAGS.weight_decay)

        biases = _bias_variable(scope + 'biases', [shape[1]])
        matmul = tf.matmul(input_, weights, name='mul')
        output = tf.add(matmul, biases, name='bias')
    return output


def full_connect(input_,
                 shape,
                 name='full_connect',
                 activation_function=None,
                 varscope=tf.get_variable_scope()):
    with tf.name_scope(name) as scope:
        z = matmul_bias(input_, shape)
        output = activation(
            z, activation_function=activation_function, varscope=varscope)
    return output


def convolution(tensor_input,
                filter_shape,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name='convolution',
                varscope=tf.get_variable_scope()):
    """
    Args:
        tensor_input:
        filter_shape: filter shape, a [height, width, output_channels, input_channels] list
        strides: stride of sliding window
        padding: padding method, eight 'VALID'(default) or 'SAME'
        name:
        varscope: variable scope for weight and bias
    Return:
        tensor
    """
    with tf.name_scope(name) as scope:
        n_input = filter_shape[0] * filter_shape[1] * \
            filter_shape[2] * filter_shape[3]
        weights = _weight_variable_with_decay(scope + 'weights', filter_shape,
                                              ncolumn=n_input, scope=varscope)
        conv_tensor = tf.nn.conv2d(tensor_input, weights, strides, padding,
                                   name="conv")
        biases = _bias_variable(
            scope + 'biases', filter_shape[3], scope=varscope)
        tensor = tf.nn.bias_add(conv_tensor, biases, name='add_bias')
    return tensor


def conv_activate(input_,
                  filter_shape,
                  strides=[1, 1, 1, 1],
                  padding='VALID',
                  name='conv_active',
                  activation_function=tf.nn.relu,
                  varscope=tf.get_variable_scope()):
    with tf.name_scope(name) as scope:
        conv = convolution(input_, filter_shape=filter_shape,
                           strides=strides, padding=padding,
                           name='convolution', varscope=varscope)
        output = activation(conv, activation_function=activation_function,
                            varscope=varscope, name='activation')
    return output


def batch_norm(input_, name=None, use_local_stat=None, is_train=None, decay=None, epsilon=None, varscope=None):
    """
    Batch normalization layer as described in:

    `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariance Shift <http://arxiv.org/abs/1502.03167>`_.

    :param input_: a NHWC or NC tensor
    :param use_local_stat: bool. whether to use mean/var of this batch or the moving average.
        Default to True in training and False in inference.
    :param decay: decay rate. default to 0.9.
    :param epsilon: default to 1e-5.
    :param varscope: scope of variable for storing mean and standard variance.
    """
    if varscope is None:
        varscope = tf.get_variable_scope()
    if name is None:
        name = "batch_norm"
    if use_local_stat is None:
        if FLAGS.is_train:
            use_local_stat = False
        else:
            use_local_stat = True
    if is_train is None:
        is_train = FLAGS.is_train
    if decay is None:
        decay = 0.9
    if epsilon is None:
        epsilon = 1e-5

    input_shape = input_.get_shape()
    n_param = input_shape[-1:]
    axis = list(xrange(len(input_shape) - 1))

    if len(input_shape) not in [2, 4]:
        raise TypeError(
            "Batch_norm only acceptes NHWC or NC tensor, got %d dim." % len(input_shape))
    n_param = input_shape[-1]  # channel number
    if n_param is None:
        raise TypeError

    with tf.name_scope(name) as scope:
        beta = _bias_variable(name='beta',
                              shape=[n_param], scope=varscope)
        gamma = _bias_variable(name='gamma',
                               shape=[n_param], scope=varscope)
        moving_mean = _bias_variable(
            'moving_mean', [n_param], scope=varscope, trainable=False)
        moving_variance = _bias_variable(
            'moving_variance', [n_param], scope=varscope, trainable=False)
        # TODO: clean implementation
        batch_mean, batch_var = tf.nn.moments(input_, axis, keep_dims=False)
        batch_mean = tf.identity(batch_mean, name="batch_mean")
        batch_var = tf.identity(batch_var, name="batch_var")
        ema = tf.train.ExponentialMovingAverage(
            decay=decay, name='exp_moving_average')
        ema_apply_op = ema.apply([batch_mean, batch_var]) # operators to apply ema
        ema_mean = ema.average(batch_mean)  # variable to store mean
        ema_var = ema.average(batch_var)  # operators to store var
        tf.add_to_collection('MOVING_AVERAGE_VARIABLES', ema_mean)
        tf.add_to_collection('MOVING_AVERAGE_VARIABLES', ema_var)
        mean, variance = tf.control_flow_ops.cond(FLAGS.is_train,
                                                  lambda: (mean, variance),
                                                  lambda: (
                                                      moving_mean, moving_variance),
                                                  name="is_train_cond")
        output = tf.nn.batch_normalization(
            input_, mean, variance, beta, gamma, FLAGS.eps, name='batch_norm')
    return output


def conv_bn_activ(input_,
                  filter_shape,
                  strides=[1, 1, 1, 1],
                  padding='VALID',
                  name='conv_active',
                  activation_function=tf.nn.relu,
                  varscope=tf.get_variable_scope()):
    with tf.name_scope(name) as scope:
        conv = convolution(input_, filter_shape=filter_shape,
                           strides=strides, padding=padding,
                           name='convolution', varscope=varscope)
        bn = batch_norm(conv, varscope=varscope)
        output = activation(bn, activation_function=activation_function,
                            varscope=varscope, name='activation')
    return output


def feature(tensor_input,
            filter_shape, strides_conv=[1, 1, 1, 1], padding_conv="SAME",
            activation_function=None,
            pooling_ksize=[1, 4, 4, 1], strides_pool=[1, 4, 4, 1], padding_pool="SAME",
            varscope=tf.get_variable_scope(),
            name='feature'):
    with tf.name_scope(name) as scope:
        conv = convolution(tensor_input, filter_shape, strides_conv,
                           padding_conv, name=scope + "convolution", varscope=varscope)
        if activation_function is None:
            acti = activation(conv, name=scope +
                              "activation", varscope=varscope)
        else:
            acti = activation(conv, activation_function,
                              name=scope + "activation", varscope=varscope)
        tensor = tf.nn.max_pool(
            acti, pooling_ksize, strides_pool, padding_pool, name=scope + "max_pooling")
    return tensor


def copy(tensor_input, shape, name='copy'):
    with tf.name_scope(name) as scope:
        buffer = tf.get_variable(scope + "buffer", shape,
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32,
                                 trainable=False)
        tensor = tf.assign(buffer, tensor_input, name='copy_input')
    return tensor


def inputs(shape, name='input'):
    """
    Shortcut to create a tf.float32 plackholder, for inputs.
    """
    input_tensor = _placeholder(name, shape)
    return input_tensor


def labels(shape, name='label'):
    """
    Shortcut to create a tf.float32 plackholder, for labels.
    """
    label_tensor = _placeholder(name, shape)
    return label_tensor


def linear(tensor_input, shape, strides=[1, 1, 1, 1], padding="SAME", name="output_layer"):
    with tf.name_scope(name) as scope:
        weights = _weight_variable(scope + 'weights', shape)
        biases = _bias_variable(scope + 'biases', [1])
        conv_tensor = tf.nn.conv2d(
            pre_tensor, weights, strides, padding, name=scope + 'conv')
        post_tensor = tf.nn.bias_add(
            conv_tensor, biases, name=scope + "add_bias")
    return post_tensor


def residual_add(original_tensor, residual_tensor, name="add_residual"):
    """
    Add residual tensor to origial tensor.    
    """
    with tf.name_scope(name) as scope:
        tensor = tf.add(residual_tensor, original_tensor, name=scope + 'add')
    return tensor


def l2_loss(inference_image, reference_image, name="loss_layer"):
    with tf.name_scope(name) as scope:
        # l2 = tf.square(inference_image - high_res_image)
        # loss = tf.sqrt(tf.reduce_sum(l2))
        loss = tf.nn.l2_loss(
            inference_image - reference_image, name=scope + name)
        tf.add_to_collection('losses', loss)
    return loss


def l2_ave_loss(inference_image, reference_image, name="loss_layer"):
    with tf.name_scope(name) as scope:
        loss = tf.nn.l2_loss(
            inference_image - reference_image, name=scope + name)
        shape = inference_image.get_shape().as_list()
        n_ele = shape[0] * shape[1] * shape[2] *shape[3]
        ave_loss = loss / n_ele
        tf.add_to_collection('losses', ave_loss)
    return loss


def psnr_loss(inference_tensor, reference_tensor, name="loss_layer"):
    with tf.name_scope(name) as scope:
        l2 = tf.square(inference_tensor - reference_tensor,
                       name='l2_difference')
        MSE = tf.reduce_mean(l2, name='MSE')
        # MSE = tf.nn.l2_loss(inference_tensor - reference_tensor, name='MSE')
        loss = tf.neg(tf.log(tf.inv(tf.sqrt(MSE + FLAGS.eps))), name='psnr')
        tf.add_to_collection('losses', loss)
    return loss


def loss_summation(name="total_loss", norm_batch=True, noramlization=1):
    with tf.name_scope(name) as scope:
        loss_list = tf.get_collection('losses')
        loss = tf.add_n(loss_list, name=scope + 'summation')            
    return loss


def crop(input_, shape, offset, num=None, name="cropping"):
    with tf.name_scope(name) as scope:
        tensor_list = tf.unpack(input_, num, name='unpack')
        tensor_crop_list = []
        for t in tensor_list:
            tc = tf.image.crop_to_bounding_box(
                t, offset[0], offset[1], shape[0], shape[1])
            tensor_crop_list.append(tc)
        tensor_cropped = tf.pack(tensor_crop_list, name='pack')
    return tensor_cropped


def down_sample(input_, filtershape, strides=[1, 2, 2, 1], name='down_sample'):
    if len(filtershape) != 2:
        raise TypeError('2D convolution window is required.')
    constant_filter = np.zeros(list(filtershape) + [1, 1])
    sz = filtershape[0] * filtershape[1]
    val = 1 / sz
    constant_filter = constant_filter * val
    filter = tf.constant(constant_filter)
    output_ = tf.nn.depthwise_conv2d(input_, filter, strides, 'SAME', name)
    return output_


def predict_loss(inference, reference, name='predic_loss'):
    with tf.name_scope(name) as scope:
        cross_enropy = tf.nn.softmax_cross_entropy_with_logits(inference,
                                                               reference,
                                                               name=scope + 'cross_entropy')
        output = tf.reduce_mean(cross_enropy, name=scope + 'loss')
    return output


def predict_accuracy(inference, reference, name='predic_accuracy'):
    """Accuracy for one-hot vector
    """
    with tf.name_scope(name) as scope:
        correct_prediction = tf.equal(tf.argmax(inference, 1), tf.argmax(
            reference, 1), name=scope + 'correct_predictions')
        output = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name=scope + 'accuracy')
    return output


def dropout(input_, keep_prob, name='dropout'):
    with tf.name_scope(name) as scope:
        output = tf.nn.dropout(input_, keep_prob)
    return output


def trainstep(loss, learn_rate, global_step, name='train_step'):
    with tf.name_scope(name) as scope:
        output = tf.train.AdamOptimizer(learn_rate).minimize(loss,
                                                             global_step,
                                                             name=name)
    return output


def trainstep_clip(loss, learn_rate, global_step, name='trainstep_clip'):
    with tf.name_scope(name) as scope:
        opt = tf.train.AdamOptimizer(learn_rate)
        grads_and_vars = opt.compute_gradients(loss)
        clipped_grad_vars = [(tf.clip_by_value(gv[0], -FLAGS.grad_clip, FLAGS.grad_clip),
                              gv[1]) for gv in grads_and_vars]
        opt.apply_gradients(clipped_grad_vars)
        train_op = opt.minimize(loss, global_step, name=name)
    return train_op
