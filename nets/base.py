""" Base class definition """

import pathlib
import tensorflow as tf
import os
import re
import numpy as np

from ..utils.general import with_config, ProgressTimer
from ..utils.prints import pp_json, pprint
from ..utils.options import Params


class Net:
    @with_config
    def __init__(self,
                 log_dir=None,
                 model_dir=None,
                 lr=1e-3,
                 is_load=False,
                 batch_size=None,
                 summary_freq=10,
                 summary_type='time',
                 save_freq=100,
                 dataset=None,
                 load_step=None,
                 ckpt_name='model.ckpt',
                 keep_prob=0.5,
                 grad_clip=1.0,
                 nb_gpus=1,
                 is_show_device_placement=False,
                 **kwargs):
        """
            data - net couple:
                data {name: np.ndarray}
                node {name: tf.tensor}
                node_name_proxy {name_in_node: name_in_data}
            task - run_op, feed_dict coutple:
                name: task name
                run_op {name: tf.tensors/tf.ops}
                feed_dict {name: node_names}
            lr - optimizer - train_step couple:
                loss {name: tf.tensor}
                optimizer {name: optimizer}
                train_step {name: tf.op}
        """
        # JSON serilizeable hyper-parameter dict.
        self.params = Params()
        self.params['log_dir'] = log_dir
        self.params['model_dir'] = model_dir
        self.params['name'] = 'Net'
        self.params['is_load'] = is_load
        self.params['summary_freq'] = summary_freq
        self.params['summary_type'] = summary_type
        self.params['load_step'] = load_step
        self.params['ckpt_name'] = ckpt_name
        self.params['save_freq'] = save_freq
        self.params['keep_prob'] = keep_prob
        self.params['batch_size'] = batch_size
        self.params['grad_clip'] = grad_clip
        self.params['nb_gpus'] = nb_gpus
        self.params['is_show_device_placement'] = is_show_device_placement
        self.params.update_short_cut()
        self.p = self.params.short_cut
        # model external nodes
        self.nodes = dict()
        self.nodes_name_proxy = dict()

        # Variables:
        #global step
        self.gs = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.__gs_value = tf.placeholder(dtype=tf.int32, name='gs_value')
        self.__gs_setter = self.gs.assign(self.__gs_value)

        # keep prob
        self.kp = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.add_node(self.kp, 'keep_prob')
        # training switch
        self.training = tf.placeholder(dtype=tf.bool, name='training')
        self.add_node(self.training, 'training')

        # learning rates
        if not isinstance(lr, dict):
            lr = {'train': lr}
        self.params['lr'] = lr
        self.lr = dict()
        with tf.name_scope('learning_rates'):
            for k in self.params['lr']:
                name = 'lr/' + k
                self.lr[k] = tf.placeholder(dtype=tf.float32, name=name)
                self.add_node(self.lr[k], name)


        # Key by sub-graph:
        self.grads = dict()
        self.losses = dict()
        self.metrices = dict()

        # Key by task
        self.optimizers = dict()
        self.train_steps = dict()
        self.summary_ops = dict()

        self.feed_dict = dict()
        self.feed_dict['default'] = ['lr/train']
        self.run_op = {
            'default': {'global_step': self.gs}
        }

        # Debug tensors
        self.debug_tensor = dict()

        # Key with 'train' and 'test'
        self.dataset = dict()

        # Session and managers
        self.sess = None
        self.saver = None
        self.summary_writer = None
        self.sv = None


# helper methods for constructing net:
    def init(self):
        self._set_model()
        self._set_train()
        self._set_saver()
        self._set_sesssv()
        self._set_summary()
        pp_json(self.params, self.params['name'] + " PARAMS:")

    def add_node(self, tensor, name=None):
        if name is None:
            name = tensor.name
        self.nodes[name] = tensor

    def _set_model(self):
        """
        Need to fill:
            self.losses
            self.grads
        """
        pass

    def train_step(self, tower_grads, opt, summary_verbose=0, name='train_step'):
        with tf.name_scope(name), tf.device('/cpu:0'):
            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over
                    # below.
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                clipv = self.params['grad_clip']
                if clipv is not None:
                    grad = tf.clip_by_value(grad, -clipv, clipv)
                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)            
            train_op = opt.apply_gradients(average_grads, global_step=self.gs)

            if summary_verbose > 0:
                for grad, var in average_grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/gradients', grad)
                    for var in tf.trainable_variables():
                        tf.summary.histogram(var.op.name, var)
        return train_op

    def _set_train(self):
        pass

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
        if self.params['is_show_device_placement']:
            config = tf.ConfigProto(log_device_placement=True)
        else:
            config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = self.sv.prepare_or_wait_for_session(config=config)

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

    def _run_helper(self, run_ops, feed_dict_keys, feed_dict_values_list):
        feed_dict = dict()
        for key in feed_dict_keys:
            key = self.nodes_name_proxy.get(key, key)
            value = None
            for d in feed_dict_values_list:
                if value is not None:
                    break
                value = d.get(key)
            if value is None:
                raise KeyError('No value of key: {} in all dicts.'.format(key))
            feed_dict[self.nodes[key]] = value
        results = self.sess.run(run_ops, feed_dict=feed_dict)
        return results

# low level api
    def partial_fit(self, data, task='train'):
        """ features are dict of mini batch numpy.ndarrays """
        feed_dict_keys = list(self.feed_dict['default'])
        feed_dict_keys += self.feed_dict['train']
        hypers = dict()
        hypers['lr/' + task] = self.params.get('lr')[task]
        hypers['keep_prob'] = self.params.get('keep_prob', 0.5)
        hypers['training'] = True
        run_op = dict(self.run_op['default'])
        run_op.update(self.run_op['train'])
        results = self._run_helper(run_op, feed_dict_keys, [hypers, data])
        return results

    def dump(self, data, task='train'):
        feed_dict_keys = list(self.feed_dict['default'])
        feed_dict_keys += self.feed_dict['train']
        hypers = dict()
        hypers['lr/' + task] = self.params.get('lr')[task]
        hypers['keep_prob'] = self.params.get('keep_prob', 0.5)
        hypers['training'] = True
        run_op = dict(self.debug_tensor)
        results = self._run_helper(run_op, feed_dict_keys, [hypers, data])
        step = self.sess.run(self.gs)
        np.save('d%d.npy' % step, result)

    def predict(self, data, **kwargs):
        """ predict a mini-batch """
        # feed_dict_keys = list(self.feed_dict['default'])
        # feed_dict_keys += self.feed_dict['predict']
        feed_dict_keys = list(self.feed_dict['predict'])
        hypers = dict()
        hypers['keep_prob'] = self.params.get('keep_prob', 1.0)
        hypers['training'] = True
        run_op = dict(self.run_op['default'])
        run_op.update(self.run_op['predict'])
        results = self._run_helper(run_op, feed_dict_keys, [hypers, data])
        return results

    def evaluate(self, data, **kwargs):
        feed_dict = dict()
        feed_dict_keys = list(self.feed_dict['default'])
        feed_dict_keys += self.feed_dict['evaluate']
        hypers = dict()
        hypers['keep_prob'] = self.params.get('keep_prob', 1.0)
        hypers['training'] = True
        run_op = dict(self.run_op['default'])
        run_op.update(self.run_op['evaluate'])
        results = self._run_helper(run_op, feed_dict_keys, [hypers, data])
        return results

    def summary(self, data, mode='train', **kwargs):
        feed_dict = dict()
        feed_dict_keys = list(self.feed_dict['default'])
        feed_dict_keys += self.feed_dict['summary']
        hypers = dict()
        for k in self.params['lr'].keys():
            hypers['lr/' + k] = self.params.get('lr')[k]
        if mode == 'train':
            hypers['keep_prob'] = self.params.get('keep_prob', 1.0)
            hypers['training'] = True
        else:
            hypers['keep_prob'] = self.params.get('keep_prob', 1.0)
            hypers['training'] = True
        run_op = dict(self.run_op['default'])
        run_op.update(self.run_op['summary'])
        results = self._run_helper(run_op, feed_dict_keys, [hypers, data])
        step = results['global_step']
        for k in self.run_op['summary'].keys():
            self.summary_writer[mode].add_summary(results[k], global_step=step)
        self.summary_writer[mode].flush()
        return results

    def reset_lr(self, name=None, lr=None, decay=10.0):
        if name is None:
            for n in self.params['lr'].keys():
                self.reset_lr(n, lr, decay)
            return None
        if lr is None:
            self.params['lr'][name] /= decay
        else:
            self.params['lr'][name] = lr
        pp_json(self.params, self.params['name'] + " PARAMS:")

    def set_dataset(self, name, dataset):
        self.dataset[name] = dataset

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

# high level api
    def train(self, steps=None, phase=1, decay=2.0):
        if not isinstance(steps, (list, tuple)):
            steps = [steps] * phase
        total_step = sum(steps)
        pt = ProgressTimer(total_step)
        cstep = 0
        for idx, sp in enumerate(steps):
            for i in range(sp):
                ss = self.dataset['train'].sample()
                res = self.partial_fit(ss)
                msg = "LOSS=%6e, STEP=%5d" % (res['loss'], res['global_step'])
                # if res['loss'] > 1e-2:
                #     self.dump(ss)
                cstep += 1
                pt.event(cstep, msg)
                if self.params['summary_type'] == 'step':
                    if i % self.params['summary_freq'] == 0:
                        self.summary_auto()
                if i % self.params['save_freq'] == 0 and i > 0:
                    self.save()
            self.save()
            if idx < len(steps) - 1:
                self.reset_lr(decay=decay)

    def predict_auto(self, data, batch_size=32, **kwargs):
        """ predict a large tensor, automatically seperate it into mini-batches. """
        nb_sample = None
        for k in data.keys():
            if nb_sample is None:
                nb_sample = data[k].shape[0]
            else:
                if nb_sample != data[k].shape[0]:
                    raise ValueError("Wrong data shape.")
        nb_blocks = nb_sample // batch_size + 1
        preds = []
        pt = ProgressTimer(nb_blocks)
        for i in range(nb_blocks):
            data_block = dict()
            for k in data.keys():
                i_start = (i - 1) * batch_size
                i_end = min([i * batch_size, nb_sample])
                if i_start >= i_end:
                    break
                data_block[k] = data[k][i_start:i_end, ...]
            preds.append(self.predict(data_block))
            pt.event(i)
        results = dict()
        for k in preds[0].keys():
            results[k] = []
        for item in preds:
            results[k].append(item[k])
        for k in results.keys():
            results[k] = np.concatenate(results[k], 0)
        return results

    def summary_auto(self):
        result = dict()
        for k in self.summary_writer.keys():
            ss = self.dataset[k].sample()
            result[k] = self.summary(ss, k)
        return result
