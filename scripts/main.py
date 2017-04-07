""" General entry for xlearn """
import matplotlib
matplotlib.use('agg')
# import argparse
# import datetime
from pathlib import Path
import shutil
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import click
import json


import xlearn.datasets
import xlearn.nets
from xlearn.utils.general import with_config, empty_list, enter_debug, ProgressTimer, config_from_dicts, print_pretty_args
from xlearn.utils.image import subplot_images


class Config(dict):
    def __init__(self, config='config.json'):
        self.config = config
        super(Config, self).__init__()

    def load(self):
        """load a JSON config file from disk"""
        filepath = Path(self.config).resolve()
        if filepath.exists():
            with open(self.config, 'r') as f:
                self.update(json.load(f))


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('--cfg', default='config.json')
@click.option('--debug', is_flag=True, default=False)
@pass_config
def xln(config, cfg, debug):
    config.config = cfg
    config.load()
    if debug:
        print("ENTER DEBUG MODE")
        enter_debug()


@xln.command()
@click.option('--net_name', '-nn', type=str)
@click.option('--filenames', '-fn', multiple=True, type=str)
def test_net_define(net_name,
                    filenames=[],
                    **kwargs):
    netc = getattr(xlearn.nets, net_name)
    net = netc(filenames=filenames)
    net.define_net()

    click.echo(net.pretty_settings())


@xln.command()
@click.option('--kind', '-k', type=str)
@click.option('--name', '-n', type=str)
@click.option('--filenames', '-fn', multiple=True, type=str)
@with_config
def cpcfg(kind, name, **kwargs):
    """ copy config files """
    print_pretty_args(cpcfg, locals())
    name += '.json'
    cfg_file = Path(os.environ['PATH_XLEARN']) / 'configs' / kind / name
    shutil.copy(cfg_file, name)


@xln.command()
@click.option('--dataset_name', '-dn', type=str)
@click.option('--filenames', '-fn', multiple=True, type=str)
@click.option('--is_save', type=bool)
@click.option('--nb_sample', default=1, type=int)
@click.option('--report_shape', is_flag=True)
def test_dataset(dataset_name,
                 filenames=[],
                 is_save=True,
                 nb_samples=32,
                 report_shape=False,
                 is_plot=False,
                 **kwargs):
    """ test datasets
    Args:
        dataset_name
        filenames
        nb_sample
        report_shape
    """
    print_pretty_args(test_dataset, locals())
    dsc = getattr(xlearn.datasets, dataset_name)
    with dsc(filenames=filenames) as dataset:
        for i in tqdm(range(kwargs['nb_sample']), ascii=True):
            s = next(dataset)
            if report_shape:
                print(len(s))
                print(len(s[0]))
                print(s[0][0].shape)
            if is_save:
                imgss = []
                for i in range(len(s[0])):
                    imgss.append(dataset.visualize(s[0][i]))
                subplot_images(imgss, is_save=True, filename='images.png')


@xln.command()
@click.option('--dataset_name', '-dn', type=str)
@click.option('--net_name', '-nn', type=str)
@click.option('--filenames', '-fn', multiple=True, type=str)
@click.option('--epochs', type=int)
@click.option('--steps_per_epoch', type=int)
@click.option('--load_step', type=int)
@with_config
def train_sr_d(dataset_name,
               net_name,
               epochs,
               steps_per_epoch,
               load_step=-1,
               filenames=[],
               **kwargs):
    """ train super resolution net
    """
    dsc = getattr(xlearn.datasets, dataset_name)
    netc = getattr(xlearn.nets, net_name)
    print_pretty_args(train_sr_d, locals())
    with dsc(filenames=filenames) as dataset:
        net_settings = {'filenames': filenames}
        if load_step is not None:
            net_settings.update({'init_step': load_step})
        net = netc(**net_settings)
        net.define_net()        
        cpx, cpy = net.crop_size
        if load_step is not None:
            if load_step > 0:
                click.secho(
                    '=' * 5 + 'LOAD PRE TRAIN WEIGHTS OF {0:7d} STEPS.'.format(load_step) + '=' * 5, fg='yellow')
                net.load(step=load_step, is_force=True)
                net.global_step = load_step
        click.secho(net.pretty_settings())
        pt = ProgressTimer(epochs * steps_per_epoch)
        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                s = next(dataset)
                loss = net.train_on_batch(
                    'sr', s[0], s[1])
                msg = "model:{0:5s}, loss={1:10e}, gs={2:7d}.".format(
                    'sr', loss, net.global_step)
                pt.event(msg=msg)
        net.save(step=net.global_step)
        net.dump_loss()


@xln.command()
@click.option('--dataset_name', '-dn', type=str)
@click.option('--net_name', '-nn', type=str)
@click.option('--filenames', '-fn', multiple=True, type=str)
@click.option('--load_step', type=int)
@with_config
def predict_sr(net_name=None,
               dataset_name=None,
               path_save='./predict',
               is_visualize=False,
               is_save=False,
               save_filename='predict.png',
               filenames=[],
               load_step=None,
               **kwargs):
    print_pretty_args(predict_sr, locals())
    dsc = getattr(xlearn.datasets, dataset_name)
    netc = getattr(xlearn.nets, net_name)

    with dsc(filenames=filenames) as dataset:
        net_settings = {'filenames': filenames}
        if load_step is not None:
            net_settings.update({'init_step': load_step})
        net = netc(**net_settings)
        net.define_net()
        click.echo(net.pretty_settings())
        if load_step is not None:
            net.load(is_force=True, step=load_step)
        net_interp = xlearn.nets.SRInterp(filenames=filenames)
        net_interp.define_net()
        s = next(dataset)
        p = net.predict('sr', s[0])
        p_it = net_interp.predict('sr', s[0])                    
        _, hr_t = net.predict('itp', s[0])
        res_sr = net.predict('res_out', s[0])
        res_it = net.predict('res_itp', s[0])
        hr = dataset.visualize(hr_t, is_no_change=True)
        lr = dataset.visualize(s[0][-1], is_no_change=True)
        sr = dataset.visualize(p, is_no_change=True)
        it = dataset.visualize(p_it, is_no_change=True)
        res_sr_l = dataset.visualize(res_sr, is_no_change=True)
        res_it_l = dataset.visualize(res_it, is_no_change=True)
        window = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
                  (-0.5, 0.5), (-0.1, 0.1), (-0.1, 0.1)]
        subplot_images((hr, lr, sr, it, res_sr_l, res_it_l), is_gray=True, size=3.0, tight_c=0.5,

                       is_save=True, filename=save_filename, window=window)

        # np.save('predict_hr.npy', s[1][0])
        # np.save('predict_lr.npy', s[0][1])
        # np.save('predict_sr.npy', p)
        # np.save('predict_res_sr.npy', res_sr)
        # np.save('predict_res_it.npy', res_it)
        res_sr_v = np.sqrt(np.mean(np.square(res_sr)))
        res_it_v = np.sqrt(np.mean(np.square(res_it)))
        print('res_sr: {0:10f}, res_it: {1:10f}'.format(
            res_sr_v, res_it_v))


@xln.command()
@click.option('--dataset_name', '-dn', type=str)
@click.option('--filenames', '-fn', multiple=True, type=str)
@click.option('--nb_images', type=int)
@click.option('--data_type', type=str, multiple=True)
@click.option('--img_filename', type=str)
@click.option('--is_save', type=bool)
@with_config
def test_dataset_image(dataset_name,
                       filenames=[],
                       nb_images=64,
                       data_type=['data', 'label'],
                       img_filename='test_dataset_image.png',
                       is_save=True,
                       **kwargs):
    if not isinstance(data_type, (list, tuple)):
        data_type = [data_type]
    nb_data_cata = len(data_type)
    dsc = getattr(xlearn.datasets, dataset_name)
    with dsc(filenames=filenames) as dataset:
        for i in range(int(np.ceil(nb_images / dataset.batch_size))):
            s = next(dataset)
            print(s[0].shape)
            print(s[1].shape)
            for i, ctype in enumerate(data_type):
                img_tensor = dataset.data_from_sample(
                    s, data_type=ctype)
                imgs = dataset.visualize(img_tensor)
                if imgs_all[i] is None:
                    imgs_all[i] = imgs
                else:
                    imgs_all[i].append(imgs)


if __name__ == '__main__':
    xln()


# D_NETS = ['SRInterp', 'SRDv0', 'SRDv1', 'SRDv1b', 'AE1D']


# class DLRun:
#     net = None
#     dataset = None

#     def __init__(self):
#         self._is_debug = False

#     def define_dataset(self, dataset_name, config_files=None, **kwargs):
#         if self._is_debug:
#             enter_debug()
#         dataset = None
#         if dataset_name == 'mnist':
#             dataset = MNIST(filenames=config_files, **kwargs)
#         elif dataset_name == 'MNIST2':
#             dataset = MNIST2(filenames=config_files, **kwargs)
#         elif dataset_name == 'celeba':
#             dataset = Celeba(filenames=config_files, **kwargs)
#         if dataset is None:
#             raise ValueError(
#                 'Unknown dataset_name {0:s}.'.format(dataset_name))
#         print(dataset.pretty_settings())
#         return dataset

#     def get_dataset(self, dataset_name):
#         dataset = None
#         if dataset_name == 'celeba':
#             dataset = Celeba
#         if dataset is None:
#             raise ValueError(
#                 'Unknown dataset_name {0:s}.'.format(dataset_name))
#         return dataset

#     def define_net(self, net_name, config_files=None):
#         if self._is_debug:
#             enter_debug()
#         net = None
#         if net_name == 'AE1D':
#             net = AE1D(filenames=config_files)
#         elif net_name == 'VAE1D':
#             net = VAE1D(filenames=config_files)
#         elif net_name == 'CVAE1D':
#             net = CVAE1D(filenames=config_files)
#         elif net_name == 'CAAE1D':
#             net = CAAE1D(filenames=config_files)
#         elif net_name == 'SRInterp':
#             net = SRInterp(filenames=config_files)
#         elif net_name == 'SRDv0':
#             net = SRDv0(filenames=config_files)
#         elif net_name == 'SRDv0b':
#             net = SRDv0b(filenames=config_files)
#         elif net_name == 'SRDv1':
#             net = SRDv1(filenames=config_files)
#         elif net_name == 'SRDv1b':
#             net = SRDv1b(filenames=config_files)
#         elif net_name == 'sr_classic':
#             net = SRClassic(filenames=config_files)
#         if net is None:
#             raise ValueError('Unknown net_name {0}.'.format(net_name))
#         print(net.pretty_settings())
#         net.define_net()
#         return net

#     @with_config
#     def train_sr(self,
#                  net_name=None,
#                  dataset_name=None,
#                  is_force_load=False,
#                  config_files=None,
#                  is_debug=False,
#                  is_reset_lr=False,
#                  lrs=None,
#                  settings=None,
#                  filenames=None,
#                  nb_epoch=32,
#                  nb_sample_epoch=32,
#                  is_p2=False,
#                  **kwargs):
#         net_name = settings.get('net_name', net_name)
#         dataset_name = settings.get('dataset_name', dataset_name)
#         is_debug = settings.get('is_debug', is_debug)
#         config_files = settings.get('config_files', config_files)
#         is_reset_lr = settings.get('is_reset_lr', is_reset_lr)
#         is_force_load = settings.get('is_force_load', is_force_load)
#         is_p2 = settings.get('is_p2', is_p2)
#         nb_epoch = settings.get('nb_epoch', nb_epoch)
#         nb_sample_epoch = settings.get('nb_sample_epoch', nb_sample_epoch)
#         lrs = settings.get('lrs', lrs)
#         print("=" * 30)
#         print("Train with setttings")
#         print(settings)
#         print("=" * 30)
#         if is_debug:
#             enter_debug()
#         net = self.define_net(net_name, config_files=config_files)
#         with self.get_dataset(dataset_name)(filenames=config_files) as dataset:
#             if is_force_load:
#                 net.load(is_force=True)
#             if is_reset_lr:
#                 net.reset_lr(lrs)
#             if net_name in D_NETS or True:
#                 pt = ProgressTimer(nb_epoch * nb_sample_epoch)
#                 for i_epoch in range(nb_epoch):
#                     loss = None
#                     for i_batch in range(nb_sample_epoch):
#                         s = next(dataset)
#                         loss_v = net.model(
#                             'sr').train_on_batch(s[0][net._nb_down_sample], s[1][0])
#                         if loss is None:
#                             loss = loss_v
#                         else:
#                             loss = 0.6 * loss + 0.4 * loss_v
#                         msg = 'loss= {0:5f}'.format(loss)
#                         pt.event(i_batch + i_epoch * nb_sample_epoch, msg)
#                     net.save('sr', 'save-%d.h5' % (i_epoch,))

#     @with_config
#     def train(self,
#               net_name=None,
#               dataset_name=None,
#               is_force_load=False,
#               config_files=None,
#               is_debug=False,
#               is_reset_lr=False,
#               lrs=None,
#               settings=None,
#               filenames=None,
#               is_p2=False,
#               **kwargs):
#         net_name = settings.get('net_name', net_name)
#         dataset_name = settings.get('dataset_name', dataset_name)
#         is_debug = settings.get('is_debug', is_debug)
#         config_files = settings.get('config_files', config_files)
#         is_reset_lr = settings.get('is_reset_lr', is_reset_lr)
#         is_force_load = settings.get('is_force_load', is_force_load)
#         is_p2 = settings.get('is_p2', is_p2)
#         lrs = settings.get('lrs', lrs)
#         print("=" * 30)
#         print("Train with setttings")
#         print(settings)
#         print("=" * 30)
#         if is_debug:
#             enter_debug()
#         net = self.define_net(net_name, config_files=config_files)
#         with self.get_dataset(dataset_name)(filenames=config_files) as dataset:
#             if is_force_load:
#                 net.load(is_force=True)
#             if is_reset_lr:
#                 net.reset_lr(lrs)
#             if net_name in D_NETS:
#                 net.model(0).fit_generator(
#                     dataset, steps_per_epoch=128, epochs=10, verbose=1,
#                     callbacks=[ModelCheckpoint(
#                         filepath=r"weightsP0.{epoch:02d}-{loss:.5f}.hdf5", period=1)]
#                 )
#                 net.save()
#                 # net.reset_lr([1e-4])
#                 # net.model_ae.fit_generator(
#                 #     dataset, steps_per_epoch=1875, epochs=40, verbose=1,
#                 #     callbacks=[ModelCheckpoint(
#                 #         filepath=r"weightsP1.{epoch:02d}-{loss:.5f}.hdf5", period=1)]
#                 # )
#                 # net.save()
#                 # net.reset_lr([1e-5])
#                 # net.model_ae.fit_generator(
#                 #     dataset, steps_per_epoch=1875, epochs=40, verbose=1,
#                 #     callbacks=[ModelCheckpoint(
#                 #         filepath=r"weightsP2.{epoch:02d}-{loss:.5f}.hdf5", period=1)]
#                 # )
#                 # net.save()
#             if net_name == 'CAAE1D':
#                 if is_p2:
#                     print('Train CAAE1D in P2')
#                     net.load()
#                     net.fit_p2(dataset, phase=0)
#                     net.save()
#                 else:
#                     print('Train CAAE1D in P1')
#                     net.fit_ae(dataset, phase=0)
#                     net.save()

#     @with_config
#     def show_mainfold(self,
#                       net_name=None,
#                       dataset_name=None,
#                       is_debug=False,
#                       config_files=None,
#                       save_filename='./mainfold.png',
#                       settings=None,
#                       **kwargs):
#         print('Genererate routine is called.')
#         net_name = settings.get('net_name', net_name)
#         dataset_name = settings.get('dataset_name', dataset_name)
#         is_debug = settings.get('is_debug', is_debug)
#         config_files = settings.get('config_files', config_files)
#         save_filename = settings.get('save_filename', save_filename)
#         if is_debug:
#             enter_debug()
#         net = self.define_net(net_name, config_files=config_files)
#         dataset = self.define_dataset(dataset_name, config_files=config_files)
#         net.load(is_force=True)
#         if net_name == 'CVAE1D' or 'CAAE1D':
#             nb_batches = 1024 // net.batch_size
#             inputs = None
#             s = next(dataset)
#             ip_c = s[0][1]
#             ip_x = s[0][0]
#             for i in range(nb_batches):
#                 s = next(dataset)
#                 ip_c = np.concatenate((ip_c, s[0][1]), axis=0)
#                 ip_x = np.concatenate((ip_x, s[0][0]), axis=0)
#             p = net.model('enc').predict(
#                 [ip_x, ip_c], batch_size=net.batch_size)
#             fig = plt.figure(figsize=(10, 10))
#             plt.plot(p[:, 0], p[:, 1], '.')
#             fig.savefig(save_filename)
#             np.save('predict_output.npy', p)

#     @with_config
#     def generate(self,
#                  net_name=None,
#                  dataset_name=None,
#                  is_debug=False,
#                  config_files=None,
#                  path_save='./generate',
#                  is_visualize=True,
#                  is_save=True,
#                  save_filename='generate.png',
#                  settings=None,
#                  **kwargs):
#         print('Genererate routine is called.')
#         net_name = settings.get('net_name', net_name)
#         dataset_name = settings.get('dataset_name', dataset_name)
#         is_debug = settings.get('is_debug', is_debug)
#         config_files = settings.get('config_files', config_files)
#         path_save = settings.get('path_save', path_save)
#         is_visualize = settings.get('is_visualize', is_visualize)
#         is_save = settings.get('is_save', is_save)
#         save_filename = settings.get('save_filename', save_filename)
#         if is_debug:
#             enter_debug()
#         net = self.define_net(net_name, config_files=config_files)
#         dataset = self.define_dataset(dataset_name, config_files=config_files)
#         net.load(is_force=True)
#         if net_name == 'CVAE1D' or 'CAAE1D':
#             s = next(dataset)
#             z = net.gen_noise()
#             # z = np.ones((net.batch_size, 2))
#             # z[:, 0] = 0.0
#             # z[:, 1] = 0.5
#             p = net.model('gen').predict(
#                 [z, s[0][1]], batch_size=net.batch_size)
#             x = dataset.visualize(s[0][1], data_type='label')
#             ae = dataset.visualize(p, data_type='data')
#             subplot_images((x[:64], ae[:64]), is_gray=True,
#                            is_save=True, filename=save_filename)
#             np.save('generate_condition.npy', s[0][1])
#             np.save('generate_input.npy', z)
#             np.save('generate_output.npy', p)


#     @with_config
#     def predict(self,
#                 net_name=None,
#                 dataset_name=None,
#                 is_debug=False,
#                 config_files=None,
#                 path_save='./predict',
#                 is_visualize=False,
#                 is_save=False,
#                 save_filename='predict.png',
#                 settings=None,
#                 filenames=None,
#                 **kwargs):
#         print("Predict routine is called.")
#         net_name = settings.get('net_name', net_name)
#         dataset_name = settings.get('dataset_name', dataset_name)
#         is_debug = settings.get('is_debug', is_debug)
#         config_files = settings.get('config_files', config_files)
#         path_save = settings.get('path_save', path_save)
#         is_visualize = settings.get('is_visualize', is_visualize)
#         is_save = settings.get('is_save', is_save)
#         save_filename = settings.get('save_filename', save_filename)
#         if is_debug:
#             enter_debug()
#         net = self.define_net(net_name, config_files=config_files)
#         dataset = self.define_dataset(dataset_name, config_files=config_files)
#         net.load(is_force=True)
#         if net_name == 'AE1D' or net_name == 'VAE1D':
#             s = next(dataset)
#             p = net.model('ae').predict(s[0], batch_size=net.batch_size)
#             if dataset_name == 'MNIST2':
#                 x = dataset.visualize(s[0], data_type='label')
#             else:
#                 x = dataset.visualize(s[0])
#             ae = dataset.visualize(p)
#             subplot_images((x[:64], ae[:64]), is_gray=True,
#                            is_save=True, filename=save_filename)
#             np.save('predict_input.npy', s[0])
#             np.save('predict_output.npy', p)
#         if net_name == 'CVAE1D' or 'CAAE1D':
#             s = next(dataset)
#             p = net.model('ae').predict(s[0], batch_size=net.batch_size)
#             xs = dataset.visualize(s[0][0], data_type='data')
#             xl = dataset.visualize(s[0][1], data_type='label')
#             ae = dataset.visualize(p, data_type='data')
#             subplot_images((xs[:64], xl[:64], ae[:64]), is_gray=True,
#                            is_save=True, filename=save_filename)
#             np.save('predict_data.npy', s[0][0])
#             np.save('predict_label.npy', s[0][1])
#             np.save('predict_output.npy', p)


#     def test_run(self, **kwargs):
#         print('Test run called.')
#         print(kwargs)


# if __name__ == "__main__":
#     fire.Fire(DLRun)
