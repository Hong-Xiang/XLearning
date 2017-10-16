""" Base class definition """
from collections import defaultdict
import pathlib
import tensorflow as tf
import os
import re
import numpy as np
from collections import ChainMap
from ..utils.general import with_config, ProgressTimer
from ..utils.prints import pp_json, pprint
from ..utils.options import Params
from tqdm import tqdm

class Net:
    """ Base class of nets.
    Which is used for:
        configs
        # TODO complete doc.
        # TODO deprecate device type, use auto inference by nb_gpus
    Level0:
        node: {node name: tensor}
        nodes_name_proxy: {node name: data name}
        hypers: {node name: values/ndarray}

        train_tasks: list of string, sub task names
        train_vars: {train sub task name: list of vars}
        losses: {train sub task name: tf.tensor}
        optimizers: {train sub task name: tf.train.xxxOptimizer}
        metrics: {train sub task name: tf.tensor}
        grads: {train sub task name: list/tuple of tf.tensors}
        train_steps: {train sub task name: tf.op}

    Level1:
        run_ops: { task names : *dict* of tf.ops to run }
        feed_dict: { task names : list of name of nodes to be feeded }
        
        run these tasks by calling run.
    Level2:
        datasets: {task name: dataset}
        data - net couple:
            data {name: np.ndarray}
            node {name: tf.tensor}
            nodes_name_proxy {name_in_node: name_in_data}
        task - run_op, feed_dict coutple:
            name: task name
            run_op {name: tf.tensors/tf.ops}
            feed_dict {name: node_names}
        lr - optimizer - train_step couple:
            loss {name: tf.tensor}
            optimizer {name: optimizer}
            train_step {name: tf.op}
    """
    @with_config
    def __init__(self,                 
                 model_dir=None,                 
                 is_load=False,
                 load_step=None,
                 save_freq=100,
                 save_type='time',
                 ckpt_name='model.ckpt',

                 log_dir=None,
                 summary_modes=('train', 'test'),
                 summary_freq=10,
                 summary_type='time',
                 summary_train=False,
                 
                 batch_size=None,

                 dataset=None,
                                  
                 keep_prob=0.5,
                 grad_clip=1.0,
                 nb_gpus=1,
                 is_bn=False,
                 is_show_device_placement=False,
                 warmup_step=100,
                 optimizer_name={'main':'Adam'},
                 lr={'main':1e-3},
                 device_type='auto',
                 **kwargs):
        """
        Inputs:
            devices_type: 'auto' or 'gpus'. If 'gpus', losses and grads are lists, [0] cpu [1-n] gpus.
        """
        # JSON serilizeable hyper-parameter dict.
        self.params = Params()
        self.params['name'] = 'Net'

        self.params['model_dir'] = model_dir
        self.params['is_load'] = is_load
        self.params['load_step'] = load_step
        self.params['log_dir'] = log_dir
        self.params['ckpt_name'] = ckpt_name
        self.params['save_type'] = save_type
        self.params['save_freq'] = save_freq

        self.params['summary_modes'] = summary_modes
        self.params['summary_freq'] = summary_freq
        self.params['summary_type'] = summary_type
        self.params['summary_train'] = summary_train
        # Commonly used configs:
        self.params['keep_prob'] = keep_prob
        self.params['batch_size'] = batch_size
        self.params['grad_clip'] = grad_clip
        self.params['nb_gpus'] = nb_gpus
        self.params['is_show_device_placement'] = is_show_device_placement
        self.params['warmup_step'] = warmup_step
        self.params['optimizer_name'] = optimizer_name
        self.params['is_bn'] = is_bn
        self.params['device_type'] = device_type

        self.params.update_short_cut()
        self.p = self.params.short_cut

        # model external nodes
        self.nodes = dict()
        self.nodes_name_proxy = dict()

        # Variables and constants (stored in hypers)
        self.hypers = dict()

        #global step
        self.gs = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        with tf.name_scope('global_step_setter'):
            self.__gs_value = tf.placeholder(dtype=tf.int32, name='gs_value')
            self.__gs_setter = self.gs.assign(self.__gs_value)
        self.add_node('global_step', tensor=self.gs, shape=[])
        
        # keep prob
        self.kp, _ = self.add_node('keep_prob', shape=[])
                
        # training switch        
        self.training, _ = self.add_node('training', shape=[], dtype=tf.bool)

        # learning rates
        if not isinstance(lr, dict):
            lr = {'main': lr}
        self.params['lr'] = lr
        self.lr = dict()
        
        for k in self.params['lr']:
            name = 'lr/'+k
            self.lr[k], _ = self.add_node(name, shape=[])
            self.hypers[name] = self.params['lr'][k]

        # train sub tasks
        self.train_tasks = ['main']
        self.train_vars = dict()
        self.losses = dict()
        self.grads = dict()        
        self.metrices = dict()
        self.optimizers = dict()
        self.train_steps = dict()
        
        

        # Key by task                
        self.summary_ops = dict()

        self.feed_dict = dict()
        self.feed_dict['default'] = list()
        for k in self.params['lr']:
            self.feed_dict['default'].append('lr/' + k)
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
        self.supervisor = None

        # Quick flags
        self.is_multi_gpu = self.p.nb_gpus > 1        

    def init(self):
        pp_json(self.params, self.params['name'] + " PARAMS:")
        self._set_model()
        self._set_train()
        self._set_saver()
        self._set_sesssv()
        self._set_summary()
        self._set_task()
        pp_json(self.run_op, "RUN OPS:")
        pp_json(self.feed_dict, "FEED DICTS:")

    def _set_model(self):
        """ Construct model(s) for net.
        To construct:
            node: {node name: tensor}
            nodes_name_proxy: {node name: data name}

            losses: {sub task name: tf.tensor or list of tf.tensors}                    
        """
        pass


    def _set_task(self):
        """ Constrcut tasks for net.
        Tasks like train, evaluate, summary, predict, etc.
        To construct:
            run_op: { task names : *dict* of tf.ops to run }
            feed_dict: { task names : list of name of nodes to be feeded }
        """
        pass
    
    def _get_optimizer(self, sub_task_name):
        optim = self.optimizers.get(sub_task_name)
        if optim is None:
            name = self.p.optimizer_name.get(sub_task_name)
            if name in ['Adam', 'adam']:
                optim = tf.train.AdamOptimizer(self.lr[sub_task_name])
            elif name in ['RMSProp', 'rmsprop']:
                optim = tf.train.RMSPropOptimizer(self.lr[sub_task_name])
        return optim

    def _get_train_vars(self, sub_task_name):
        vars_list = self.train_vars.get(sub_task_name)
        return vars_list
        
    def _set_train(self):
        """ Construct train steps.
        After calling this method, all train_steps[train_tasks.kyes()] should be constructed.
        
        Automatically construct train/sub_task tasks with sub_task in train_tasks.        
        Be careful on naming:
            train_tasks: list of sub_task names
            losses: dict of losses, value can be scalar(single gpu) or list of scalar(multi gpus)
            optimizers: dict of optimizers 
        
        Generate:
            grads
            train_steps

        """
        #TODO: seperate optimizer getter.
        for k in self.train_tasks:
            with tf.name_scope('train_step_'+k):
                optim = self._get_optimizer(k)
                var_list = self._get_train_vars(k)                
                if self.is_multi_gpu:                                    
                    self.grads = defaultdict(list)
                    for i in range(self.p.nb_gpus):
                        with tf.device('/gpu:%d' % i):
                            self.grads[k].append(optim.compute_gradients(self.losses[k][i], var_list))
                        # TODO: test perfomance of train step on cpu/gpus
                    with tf.device('/cpu:0'):                        
                        self.train_steps[k] = self.train_step(self.grads[k], optim, self.p.summary_train)
                else:
                    self.grads[k] = optim.compute_gradients(self.losses[k], var_list)                
                    self.train_steps[k] = self.train_step([self.grads[k]], optim, self.p.summary_train)

    
    def _set_saver(self):
        #TODO: partial saver and loader.
        self.saver = tf.train.Saver()
        pass

    # def _set_sesssv(self):
    #     # sv_para = {'summary_op': None}
    #     # sms = self.params.get('save_model_secs')
    #     # if sms is not None:
    #     #     sv_para['save_model_secs'] = sms
    #     # if self.params['load_step'] is not None:
    #     #     sv_para['init_fn'] = self.load
    #     # sv_para['saver'] = self.saver
    #     # self.supervisor = tf.train.Supervisor(**sv_para)
    #     if self.params['is_show_device_placement']:
    #         config = tf.ConfigProto(log_device_placement=True)
    #     else:
    #         config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     self.sess = tf.Session(config=config)
    #     # self.saver = self.supervisor.saver

    def _set_sesssv(self):
        sv_para = {'summary_op': None}
        sms = self.params.get('save_model_secs')
        if sms is not None:
            sv_para['save_model_secs'] = sms
        if self.params['load_step'] is not None:
            sv_para['init_fn'] = self.load
        sv_para['saver'] = self.saver
        self.supervisor = tf.train.Supervisor(**sv_para)
        if self.params['is_show_device_placement']:
            config = tf.ConfigProto(log_device_placement=True)
        else:
            config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = self.supervisor.prepare_or_wait_for_session(config=config)
        # self.saver = self.supervisor.saver

    def _set_summary(self):
        self.summary_writer = dict()
        for k in self.p.summary_modes:
            path_log = str(pathlib.Path(self.params['log_dir']) / k)        
            self.summary_writer[k] = tf.summary.FileWriter(path_log, self.sess.graph)        

        # Add summary service
        if self.params['summary_type'] == 'time':
            self.supervisor.loop(self.params['summary_freq'], self.summary_auto)
    

    


    # Helper functions for constructing net:
    def add_node(self, name=None, shape=None, tensor=None, is_multi_gpu=False, proxy_name=None, dtype=None, verbose=0):
        """Add node for net.
            All placeholders should be added via add_node.
            Some tensors can be add to nodes if they are feedable.
        Inputs:
            name: node name, if is None, use tensor.name
            shape: node shape/tensor shape, if is `None`, use tensor.shape
            tensor: tensor to add to nodes. If is `None`, a placeholder will be created.
            is_multi_gpu: split tensor into multiple partial mini-batches.
        Returns:
            tensor: main part of tensor
            tensors_gpu: list of splited tensors
        Raises:
            ValueError
        """
        if name is None:
            if tensor is None:
                raise ValueError("Name cannot be None if tensor is None.")
            else:
                name = tensor.name

        if shape is None:
            shape = tensor.shape.as_list()
        
        if dtype is None:
            if tensor is not None:
                dtype = tensor.dtype
            else:
                dtype = tf.float32
        # Check compatiblity
        if tensor is not None:
            tensor_shape = tensor.shape
            if len(tensor_shape) > 0:            
                if tensor_shape is not None:
                    tensor_shape = tensor_shape.as_list()            
                if tensor_shape[0] is None:
                    tensor_shape[0] = -1
                shape = list(shape)
                if shape[0] is None:
                    shape[0] = -1
            if not tensor_shape == shape:
                if not (len(tensor_shape) == 0 and shape == [1]):
                    raise ValueError("shape {0} differnt to tensor {1}.".format(shape, tensor))

        with tf.name_scope(name):
            if tensor is None:
                if self.p.device_type == 'auto':
                    tensor = tf.placeholder(dtype=dtype, shape=shape, name='main')
                elif self.p.device_type == 'gpus':
                    with tf.device('/cpu:0'):
                        tensor = tf.placeholder(dtype=dtype, shape=shape, name='main')
            self.nodes[name] = tensor                
            if proxy_name is not None:
                self.nodes_name_proxy[name] = proxy_name        

            if is_multi_gpu:
                batch_size_gpu = self.p.batch_size // self.p.nb_gpus
                shape_gpu = list(shape)
                shape_gpu[0] = batch_size_gpu
                tensors_gpu = []
                for i in range(self.p.nb_gpus):                    
                    name_part = 'part_%d'%i
                    device = '/gpu:%d'%i
                    with tf.device(device):
                        tensors_gpu.append(tf.slice(tensor, [i*batch_size_gpu, 0, 0, 0], shape_gpu, name=name_part))
            else:
                tensors_gpu = [tensor]
        return tensor, tensors_gpu
    


    def train_step(self, tower_grads, opt, is_summary=False, name='train_step'):
        """
        Inputs:
            tower_grads: *list or tuple* of grads
            opt: optimizer
        """
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

            if is_summary > 0:
                for grad, var in average_grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/gradients', grad)
                    for var in tf.trainable_variables():
                        tf.summary.histogram(var.op.name, var)
        return train_op



    def reset_lr(self, name=None, lr=None, decay=10.0):
        """ reset learning rate by value or decay.
        """
        if name is None:
            for n in self.params['lr'].keys():
                self.reset_lr(n, lr, decay)
            return None
        if lr is None:
            self.params['lr'][name] /= decay
        else:
            self.params['lr'][name] = lr
        pp_json(self.params, self.params['name'] + " PARAMS:")

    def get_global_step(self):
        return self.sess.run(self.gs)

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

    # level-1 api
    def run(self, task, datas=None, verbose=0):
        """
        """
        run_ops = self.run_op[task]
        true_feed_dict = list(self.feed_dict['default'])
        true_feed_dict += self.feed_dict[task]        
        if verbose >= 1:        
            pp_json({'run_ops': run_ops, 'to_feed': true_feed_dict}, "RUN TASK: "+task)
        feed_dict = dict()
        for k in true_feed_dict:
            k_true = self.nodes_name_proxy.get(k, k)
            datas_all = ChainMap(datas, self.hypers)
            feed_dict[self.nodes[k]] = datas_all.get(k_true)
            if feed_dict[self.nodes[k]] is None:
                msg_format = r"Node: {0} with proxy name {1} was not found in hyers {2} and datas {3}"
                msg_content = (k, k_true, list(self.hypers.keys()), list(datas.keys()))
                raise ValueError(msg_format.format(*msg_content))
        out = self.sess.run(run_ops, feed_dict=feed_dict)
        return out

    def partial_fit(self, data, sub_task=None, verbose=0):
        """ features are dict of mini batch numpy.ndarrays """        
        task = 'train'
        if sub_task is not None:
            task = '{0}/{1}'.format(task, sub_task)
        true_data = dict()
        true_data['training'] = True
        true_data['keep_prob'] = self.p.keep_prob
        true_data.update(data)
        return self.run(task, true_data, verbose)
    
    def predict(self, data, sub_task=None, verbose=0):
        """ predict a mini-batch """
        task = 'predict'
        if sub_task is not None:
            task = '{0}/{1}'.format(task, sub_task)
        true_data = dict()
        true_data['training'] = False
        true_data['keep_prob'] = 1.0
        true_data.update(data)
        return self.run(task, true_data, verbose)

    def evaluate(self, data, sub_task=None, verbose=0):
        task = 'evaluate'
        if sub_task is not None:
            task = '{0}/{1}'.format(task, sub_task)
        true_data = dict()
        true_data['training'] = False
        true_data['keep_prob'] = 1.0
        true_data.update(data)
        return self.run(task, true_data, verbose)

    def dump(self, task, datas, save_name=None, verbose=0):
        result = self.run(task, datas, verbose)
        if save_name is None:
            save_name = 'dump%d'%(self.get_global_step())
        np.savez(save_name, **result)

    def summary(self, data, mode=None, sub_task=None, verbose=0):
        if mode is None:
            mode = 'train'
        task = 'summary'
        if sub_task is not None:
            task = '{0}/{1}'.format(task, sub_task)
        true_data = dict()
        true_data['training'] = False
        true_data['keep_prob'] = 1.0                
        true_data.update(data)
        results = self.run(task, true_data, verbose)
        step = self.get_global_step()
        for k in self.run_op['summary'].keys():
            self.summary_writer[mode].add_summary(results[k], global_step=step)
        self.summary_writer[mode].flush()
        return results

    def set_dataset(self, name, dataset):
        self.dataset[name] = dataset


    #level-2 api
    def train_kernel(self, sub_task):
        ss = next(self.dataset[sub_task])
        res = self.partial_fit(ss, sub_task)
        msg = "LOSS=%6e, STEP=%5d" % (res['loss'], res['global_step'])
        return res, msg

    def set_dataset_auto(self, dataset_train, dataset_test):
        self.dataset['main'] = dataset_train
        self.dataset['train'] = dataset_train
        self.dataset['test'] = dataset_test

    def _train_sub_task_schadule(self, sub_task=None):
        if sub_task is None:
            sub_task = 'main'
        while True:
            yield sub_task

    def train(self, sub_task=None, steps=None, phase=1, decay=2.0, warmup=False):
        """ high level train.
        Inputs:
            sub_task: (Optional), if not None, run task train/sub_task
            steps: int or list of ints, steps per phase.
            phase: int. If steps is int, construct steps = [steps] * phase
            decay: learning rate decay per phase
            TODO: check warmup.
        """
        if not isinstance(steps, (list, tuple)):
            steps = [steps] * phase
        total_step = sum(steps)

        
        task_gen = self._train_sub_task_schadule(sub_task)

        if warmup:
            # warmup
            pt = ProgressTimer(total_step)
            cstep = 0
            lr_bak = self.p.lr['main']
            warming_up_phase = 10
            self.reset_lr(decay=2.0**warming_up_phase)
            pp_json(self.params, self.params['name'] + " PARAMS:")            
            warmup_step = self.params['warmup_step']
            for i in range(warming_up_phase):
                for j in range(warmup_step):
                    _, msg = self.train_kernel(next(task_gen))
                    cstep += 1
                    pt.event(cstep, "[WARM]"+msg)
                self.reset_lr(decay=0.5)

        pt = ProgressTimer(total_step)
        cstep = 0
        for idx, sp in enumerate(steps):
            for i in range(sp):
                if warmup and warming_up > 0:
                    if warmup_step <= 0:
                        
                        warming_up -= 1
                        warmup_step = warmup_step = self.params['warmup_step']
                    else:
                        warmup_step -= 1
                
                # if res['loss'] > 1e-2:
                #     self.dump(ss)
                cstep += 1
                _, msg = self.train_kernel(next(task_gen))
                pt.event(cstep, msg)
                if self.params['summary_type'] == 'step':
                    if i % self.params['summary_freq'] == 0 and i > 0:
                        self.summary_auto()
                if i % self.params['save_freq'] == 0 and i > 0:
                    self.save()
            
            # force save at end of each phase
            self.save()

            # decay lr at end of phase except the last one.
            if not idx == len(steps) - 1:
                self.reset_lr(decay=decay)

    def predict_auto(self, data, sub_task=None, batch_size=None, **kwargs):
        """ predict a large tensor, automatically seperate it into mini-batches. """
        nb_sample = None
        if batch_size is None:
            batch_size = self.p.batch_size
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
            skip = False
            data_block = dict()
            for k in data.keys():
                i_start = i * batch_size
                i_end = min([(i + 1) * batch_size, nb_sample])                
                if i_start >= i_end:
                    skip = True
                    break
                data_block[k] = data[k][i_start:i_end, ...]
            if not skip:
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
            result[k] = self.summary(ss, mode=k)
        return result
