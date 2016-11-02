from __future__ import absolute_import, division, print_function
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _weight_variable(name, shape, stddev = FLAGS.stddev, scope=None):
    """Helper to create a Variable
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    initer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    #initer = tf.constant_initializer(1.0)
    if scope is None:
        var = tf.get_variable(name, shape=shape, initializer=initer, dtype=dtype)
    else:
        with 
        var = tf.get_variable(name, shape=shape, initializer=initer, dtype=dtype)
    return var

def _weight_variable_with_decay(name, shape, stddev, wd):
    """Helper to create a Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = weight_variable(name, shape, stddev)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name+'/weight_loss')

        #WARNNNING: How does the following line work is not sure yet.
        tf.add_to_collection('losses', weight_decay)
        #============================================================

    return var

def _bias_variable(name, shape):
    dtype = tf.float32
    initer = tf.constant_initializer(0.0)
    var = tf.get_variable(name, shape, initializer=initer, dtype=dtype)
    return var

def _placeholder(name, shape):
    dtype = tf.float32
    var = tf.placeholder(dtype, shape=shape, name=name)
    return var


def full_connect_layer(pre_tensor, shape, name = None):
    with tf.name_scope(name) as scope:
        weights = _weight_variable_with_decay(scope+'/weights',
                                              [pre_unit, post_unit],
                                              1.0/np.sqrt(pre_unit),
                                              WEIGHT_DECAY_RATIO)

        biases = _bias_variable(scope+'/biases', [post_unit])
        #post_tensor = tf.nn.tanh(tf.matmul(pre_tensor, weights)+biases)
        post_tensor = tf.nn.relu6(tf.nn.bias_add(tf.matmul(pre_tensor,
                                                           weights),
                                                 biases),
                                  name="relu")
    return post_tensor

def conv_layer(pre_tensor, shape, strides = [1, 1, 1, 1], padding = "VALID", name = None):
    with tf.name_scope(name) as scope:        
        weights = _weight_variable(scope+'weights', shape)
        conv_tensor = tf.nn.conv2d(pre_tensor, weights, strides, padding,
                                   name=scope+"conv")
        biases = _bias_variable(scope+'biases', shape[3])
        post_tensor = tf.nn.relu(tf.nn.bias_add(conv_tensor, biases),
                                 name=scope+"relu")
    return post_tensor

def max_pool_layer(pre_tensor, ksize, strides = [1,4,4,1], padding = "VALID", name = None):
    with tf.name_scope(name) as scope:
        post_tensor = tf.nn.max_pool(pre_tensor, ksize, strides, padding,
                                     name=scope + "/max_pooling")
    return post_tensor

def full_conv_layer(pre_tensor, shape, ksize, strides_conv = [1,1,1,1], padding_conv = "SAME", \
        strides_pool = [1,4,4,1], padding_pool = "VALID", name = None):
    with tf.name_scope(name) as scope:
        conv_tensor = conv_layer_new(pre_tensor, shape, strides_conv, padding_conv, name = scope+"/conv")
        post_tensor = max_pool_layer(conv_tensor, ksize, strides_pool, padding_pool, name = scope +"/pool")
    return post_tensor

def indentity_layer(pre_tensor, name = None):
    with tf.name_scope(name) as scope:    
        dtype = tf.float32
        initer = tf.constant_initializer(0.0)
        buffer = tf.get_variable(scope+"buffer", [FLAGS.batch_size, FLAGS.height, FLAGS.width, 1], initializer=initer, dtype=dtype)
        post_tensor = tf.assign(buffer, pre_tensor)
    return post_tensor 

def input_layer(shape, name = None):
    input_tensor = _placeholder(name, shape)
    return input_tensor

def label_layer(shape, name = None):
    label_tensor = _placeholder(name, shape)
    return label_tensor

def output_layer(pre_tensor, shape, strides = [1, 1, 1, 1], padding="SAME", name="output_layer"):
    with tf.name_scope(name) as scope:
        weights = _weight_variable(scope+'weights', shape)
        biases = _bias_variable(scope+'biases', [1])
        conv_tensor = tf.nn.conv2d(pre_tensor, weights, strides, padding, name=scope+'conv')
        post_tensor = tf.nn.bias_add(conv_tensor, biases, name=scope+"add_bias")
    return post_tensor

def recon_layer(residual_image, low_res_image, name = "recon_layer"):
    with tf.name_scope(name) as scope:
        post_tensor = tf.add(residual_image, low_res_image, name = scope + 'add')
    return post_tensor

def l2_loss_layer(inference_image, high_res_image, name = "loss_layer"):
    with tf.name_scope(name) as scope:
        l2 = tf.square(inference_image - high_res_image)
        loss_tensor = tf.sqrt(tf.reduce_sum(l2))
        tf.scalar_summary('loss',loss_tensor)
    return loss_tensor

def psnr_loss_layer(inference_tensor, reference_tensor, name = "loss_layer"):
    with tf.name_scope(name) as scope:
        l2 = tf.square(inference_tensor - reference_tensor)
        MSE = tf.reduce_sum(l2)
        loss_tensor = tf.neg(tf.log(tf.inv(tf.sqrt(MSE))))
        tf.scalar_summary('loss',loss_tensor)
    return loss_tensor

def crop_layer(pre_tensor, shape, offset, name="crop_layer"):
    with tf.name_scope(name) as scope:
        tensor_list = tf.unpack(pre_tensor, name=scope+'unpack')
        tensor_crop_list = []
        for t in tensor_list:
            tc = tf.image.crop_to_bounding_box(t, offset[0], offset[1], shape[0], shape[1])
            tensor_crop_list.append(tc)
        tensor_cropped = tf.pack(tensor_crop_list, name=scope+'pack') 
    return tensor_cropped
