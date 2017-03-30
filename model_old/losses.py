import tensorflow as tf

def loss_lsgan(logit_true, logit_fake, batch_size, a=-1.0, b=1.0, c=0.0, is_summary=True, name='loss_lsgan'):
    with tf.name_scope(name):
        with tf.name_scope('loss_gen'):
            loss_gen = tf.reduce_mean(tf.square(logit_fake - c))
        with tf.name_scope('loss_cri'):
            loss_cri = tf.reduce_mean(
                tf.square(logit_fake - a) + tf.square(logit_true - b))
        if is_summary:
            tf.summary.scalar(name+'/loss_gen', loss_gen)
            tf.summary.scalar(name+'/loss_cri', loss_cri)
    return loss_gen, loss_cri
