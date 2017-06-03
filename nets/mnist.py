import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from ..utils.general import with_config
from .base import Net
from ..models.image import conv2d

class MNISTRecon0(Net):
    @with_config
    def __init__(self,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['name'] = "MNISTRecon0"

    def _set_model(self):
        data, _ = self.add_node('data', shape=[None, 28, 28, 1])
        label, _ = self.add_node('label', shape=[None, 1], dtype=tf.int32)
        tf.summary.image('data', data)
        h = data
        with arg_scope([conv2d], activation=tf.nn.elu):
            h = conv2d(h, 32, 5)
            h = conv2d(h, 64, 1, strides=(2, 2), padding='valid')        
            h = conv2d(h, 64, 5)
            h = conv2d(h, 64, 1, strides=(2, 2), padding='valid')
            h = tf.reshape(h, [-1, 7 * 7 * 64])
        h = tf.layers.dense(h, 1024, activation=tf.nn.relu)
        h = tf.nn.dropout(h, self.kp)
        y_pred = tf.layers.dense(h, 10)
        self.add_node('y_pred', tensor=y_pred)
        self.add_node('dig_pred', tensor=tf.argmax(y_pred, 1))
        y_label = self.nodes['label']
        y_one_hot = tf.one_hot(y_label, 10)
        y_one_hot = tf.reshape(y_one_hot, [-1, 10])

        self.losses['main'] = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=y_one_hot, logits=y_pred)
        tf.summary.scalar('loss', self.losses['main'])

        _, accuracy = tf.metrics.accuracy(self.nodes['label'], self.nodes['dig_pred'])
        self.metrices['main'] = accuracy
        tf.summary.scalar('acc', accuracy)

        self.summary_ops['main'] = tf.summary.merge_all()
    
    def _set_task(self):
        """ Constrcut tasks for net.
        Tasks like train, evaluate, summary, predict, etc.
        To construct:
            run_ops: { task names : *dict* of tf.ops to run }
            feed_dict: { task names : list of name of nodes to be feeded }
        """
        self.run_op.update({
            'train/main': {
                'train': self.train_steps['main'],
                'loss': self.losses['main'],
                'global_step': self.gs                
            },

            'predict': {
                'digit': self.nodes['dig_pred']
            },

            'summary': {
                'summary': self.summary_ops['main']
            }
        })

        self.feed_dict.update({
            'train/main': ['data', 'label', 'keep_prob'],
            'summary': ['data', 'label', 'keep_prob'],
            'predict': ['data']
        })