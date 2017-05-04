""" mnist recognition """
import tensorflow as tf



    
def mnist_recon(features, labels, mode, params, config):
    h = features['data']
    h = tf.layers.conv2d(h, 32, 5, activation=tf.nn.relu)
    h = tf.nn.max_pool(h, 2, 2)
    h = tf.layers.conv2d(h, 64, 5, activation=tf.nn.relu)
    h = tf.nn.max_pool(h, 2, 2)
    h = tf.reshape(h, [-1, 7*7*64])
    h = tf.layers.dense(h, 1024, activation=tf.nn.relu)
    if mode == tf.estimator.ModeKeys.TRAIN:
        h = tf.nn.dropout(h, 0.5)
    y_pred = tf.layers.dense(h, 10, activation=tf.nn.sigmoid)
    pred_dig = tf.argmax(y_pred, 1)
    y_label = labels['label']
    y_one_hot = tf.one_hot(y_label, 10)
    loss = tf.losses.softmax_cross_entropy(y_one_hot, y_pred)
    train_step = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.metrics.accuracy(y_one_hot, y_pred)
    return tf.estimator.EstimatorSpec(mode, pred_dig, loss, train_step, {'acc': accuracy})

