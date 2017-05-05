import pathlib
import tensorflow as tf
import os
import re
import numpy as np

from ..utils.general import with_config, ProgressTimer
from ..utils.prints import pp_json, pprint


class Net:
    @with_config
    def __init__(self,
                 log_dir=None,
                 model_dir=None,
                 lr=None,
                 is_load=False,
                 summary_freq=10,
                 summary_type='time',
                 save_freq=100,
                 dataset=None,
                 load_step=None,
                 ckpt_name='model.ckpt',
                 ** kwargs):
        # JSON serilizeable hyper-parameter dict.
        self.params = dict()
        self.params['log_dir'] = log_dir
        self.params['model_dir'] = model_dir
        self.params['name'] = 'Net'
        self.params['is_load'] = is_load
        self.params['summary_freq'] = summary_freq
        self.params['summary_type'] = summary_type
        self.params['load_step'] = load_step
        self.params['ckpt_name'] = ckpt_name
        self.params['save_freq'] = save_freq
        # Supervisor and managed session
        self.sv = None
        self.sess = None

        # Ops
        self.input = dict()
        self.output = dict()
        self.label = dict()
        self.loss = dict()
        self.train_op = dict()
        self.metric = dict()
        self.summary_op = dict()
        # Variables:
        #global step
        self.gs = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.__gs_value = tf.placeholder(dtype=tf.int32, name='gs_value')
        self.__gs_setter = self.gs.assign(self.__gs_value)
        self.feed_dict = dict()
        self.run_op = {'global_step': self.gs}
        self.kp = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.params['lr'] = lr
        self.lr = dict()
        for k in self.params['lr']:
            self.lr[k] = tf.placeholder(dtype=tf.float32, name=k)
            self.feed_dict.update({self.lr[k]: self.params['lr'][k]})
            tf.summary.scalar('lr/' + k, self.lr[k])
        self.dataset = None
        self.saver = None
        self.summary_writer = None

    def reset_lr(self, lr=None, decay=10.0):
        for k in self.lr.keys():
            if lr is None:
                self.params['lr'][k] /= decay
                self.feed_dict.update({self.lr[k]: self.params['lr'][k]})
            else:
                self.feed_dict.update({self.lr[k]: lr[k]})
        pp_json(self.params, self.params['name'] + " PARAMS:")

    def set_dataset(self, dataset):
        self.dataset = dataset

    def init(self):
        self._set_model()
        self._set_saver()
        self._set_sesssv()
        self._set_summary()
        pp_json(self.params, self.params['name'] + " PARAMS:")

    def save(self):
        path_save = pathlib.Path(self.params['model_dir'])
        if not path_save.is_dir():
            os.mkdir(path_save)
        path_save = str(path_save / self.params['ckpt_name'])
        step = self.sess.run(self.gs)
        pprint("[SAVE] model with step: {} to path: {}.".format(step, path_save))
        self.saver.save(self.sess, path_save, global_step=step)

    def load(self, sess):
        path_load = pathlib.Path(self.params['model_dir'])
        step = self.params['load_step']
        if step == -1:
            if not path_load.is_dir():
                return
            pattern = self.params['ckpt_name'] + '-' + '([0-9]+)' + '-*'
            p = re.compile(pattern)
            for f in path_load.iterdir():
                mat = p.match(f.name)
                if mat and step < int(mat[1]):
                    step = int(mat[1])
        path_load = str(path_load / self.params['ckpt_name'])
        path_load += '-%d' % (step)
        pprint("[LOAD] model from: {}.".format(path_load))
        self.saver.restore(sess, path_load)
        sess.run(self.__gs_setter, feed_dict={self.__gs_value: step})

    def _set_model(self):
        pass

    def _set_summary(self):
        self.summary_writer = dict()
        path_log_train = str(pathlib.Path(self.params['log_dir']) / 'train')
        path_log_test = str(pathlib.Path(self.params['log_dir']) / 'test')
        self.summary_writer['train'] = tf.summary.FileWriter(
            path_log_train, self.sess.graph)
        self.summary_writer['test'] = tf.summary.FileWriter(
            path_log_test, self.sess.graph)
        if self.params['summary_type'] == 'time':
            if self.dataset is None:
                raise TypeError(
                    'summary_type is time and no dataset is given.')
            self.sv.loop(self.params['summary_freq'], self.summary_auto)

    def _set_saver(self):
        self.saver = tf.train.Saver()

    def _set_sesssv(self):
        sv_para = {'summary_op': None}
        sms = self.params.get('save_model_secs')
        if sms is not None:
            sv_para['save_model_secs'] = sms
        if self.params['load_step'] is not None:
            sv_para['init_fn'] = self.load
        self.sv = tf.train.Supervisor(**sv_para)
        self.sess = self.sv.prepare_or_wait_for_session()

    def partial_fit(self, data, **kwargs):
        """ features are dict of mini batch numpy.ndarrays """
        feed_dict = dict()
        for k in self.input.keys():
            feed_dict.update({self.input[k]: data[k]})
        for k in self.label.keys():
            feed_dict.update({self.label[k]: data[k]})
        train_kp = self.params.get('keep_prob', 0.5)
        feed_dict.update({self.kp: train_kp})
        feed_dict.update(self.feed_dict)
        run_op = dict()
        run_op.update(self.run_op)
        run_op.update(self.train_op)
        run_op.update(self.loss)
        results = self.sess.run(run_op, feed_dict=feed_dict)
        return results

    def train(self, steps=None, phase=1, decay=2.0):
        pt = ProgressTimer(steps)
        steps_phase = int(np.ceil(steps // phase))
        cstep = 0
        for _ in range(phase):
            for i in range(steps_phase):
                ss = self.dataset['train'].sample()
                res = self.partial_fit(ss)
                msg = "LOSS=%6e, STEP=%5d" % (res['loss'], res['global_step'])
                cstep += 1
                pt.event(cstep, msg)
                if self.params['summary_type'] == 'step':
                    if i % self.params['summary_freq'] == 0:
                        self.summary_auto()
                if i % self.params['save_freq'] == 0 and i > 0:
                    self.save()
            self.save()
            self.reset_lr(decay=decay)

    def predict(self, data, **kwargs):
        feed_dict = dict()
        for k in self.input.keys():
            feed_dict.update({self.input[k]: data[k]})
        feed_dict.update(self.feed_dict)
        feed_dict.update({self.kp: 1.0})
        run_op = dict()
        run_op.update(self.run_op)
        run_op.update(self.output)
        results = self.sess.run(run_op, feed_dict=feed_dict)
        return results

    def evaluate(self, data, **kwargs):
        feed_dict = dict()
        for k in self.input.keys():
            feed_dict.update({self.input[k]: data[k]})
        for k in self.label.keys():
            feed_dict.update({self.label[k]: data[k]})
        feed_dict.update(self.feed_dict)
        feed_dict.update({self.kp: 1.0})
        run_op = dict()
        run_op.update(self.run_op)
        run_op.update(self.loss)
        run_op.update(self.metric)
        results = self.sess.run(run_op, feed_dict=feed_dict)
        return results

    def summary(self, data, mode='train', **kwargs):
        feed_dict = dict()
        for k in self.input.keys():
            feed_dict.update({self.input[k]: data[k]})
        for k in self.label.keys():
            feed_dict.update({self.label[k]: data[k]})
        feed_dict.update(self.feed_dict)
        feed_dict.update({self.kp: 1.0})
        run_op = dict()
        run_op.update(self.summary_op)
        run_op.update({'global_step': self.gs})
        results = self.sess.run(run_op, feed_dict=feed_dict)
        step = results.pop('global_step')
        for k in results.keys():
            self.summary_writer[mode].add_summary(results[k], global_step=step)
        self.summary_writer[mode].flush()
        return results

    def summary_auto(self):
        result = dict()
        for k in self.summary_writer.keys():
            ss = self.dataset[k].sample()
            result[k] = self.summary(ss, k)
        return result
