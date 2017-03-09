from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from xlearn.dataset import MNIST
from xlearn.nets.gans import LSGAN
from xlearn.utils.image import subplot_images
from xlearn.utils.general import enter_debug, ProgressTimer


enter_debug()


def test(cfs):
    print("Test Routine with config files:", cfs)
    net = LSGAN(filenames=cfs)    
    net.define_net()


def train(nb_batches, cfs):
    dataset = MNIST(filenames=cfs)
    print("=" * 30)
    print("DATASET SETTINGS:")
    print(dataset.pretty_settings())
    print("=" * 30)
    net = LSGAN(filenames=cfs)
    net.define_net()
    print("=" * 30)
    print("NETWORK SETTINGS:")
    print(net.pretty_settings())
    print("=" * 30)
    ptp = ProgressTimer(net.pre_train)
    for i in range(net.pre_train):
        s = next(dataset)
        z = net.gen_latent()
        loss_c = net.train_on_batch('Cri', [s[0], z], [])
        msg = 'loss_c= %f' % loss_c
        ptp.event(i, msg)
    pt = ProgressTimer(nb_batches)
    loss_c = np.nan
    loss_g = np.nan
    for i in range(nb_batches):
        s = next(dataset)
        z = net.gen_latent()
        if i % net.gen_freq > 0:
            loss_c = net.train_on_batch('Cri', [s[0], z], [])
            msg = 'c_step, loss_c= %f; loss_g= %f' % (loss_c, loss_g)
            pt.event(i, msg)
        else:
            loss_g = net.train_on_batch('WGan', [s[0], z], [])
            msg = 'g_step, loss_c= %f; loss_g= %f' % (loss_c, loss_g)
            pt.event(i, msg)
        if i % 1000 == 0:
            net.save('AutoEncoder')
    net.save('AutoEncoder', is_print=True)


def predict(cfs):
    dataset = MNIST(filenames=cfs)
    net = LSGAN(filenames=cfs)
    net.define_net()
    print(net.pretty_settings())
    net.load('AutoEncoder')
    x = net.gen_latent()
    p = net.predict('Decoder', [x])
    imgs = dataset.visualize(p[0])
    subplot_images((imgs, ), is_gray=True, size=3.0, tight_c=0.5)


def show_mainfold(cfs):
    dataset = MNIST(filenames=cfs)
    nb_axis = int(np.sqrt(dataset._batch_size))
    x = np.linspace(-1.5, 1.5, nb_axis)
    y = np.linspace(-1.5, 1.5, nb_axis)
    pos = np.meshgrid(x, y)
    xs = pos[0]
    ys = pos[1]
    xs = xs.reshape([-1])
    ys = ys.reshape([-1])
    net = LSGAN(filenames=cfs)
    net.define_net()
    net.load('Gen')
    latents = np.array([xs, ys]).T
    p = net.predict('Gen', [latents])
    imgs = dataset.visualize(p[0])
    subplot_images((imgs, ), nb_max_row=nb_axis,
                   is_gray=True, size=1.0, tight_c=0.5)

if __name__ == "__main__":
    task = sys.argv[1]
    config_files = sys.argv[2:]
    print("Running with task:", task)
    if task == 'train':
        nb_batches = int(sys.argv[2])
        config_files = sys.argv[3:]
        train(nb_batches, config_files)
    elif task == 'predict':
        predict(config_files)
    elif task == 'show':
        show_mainfold(config_files)
    elif task == 'test':
        test(config_files)
