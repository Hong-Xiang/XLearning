from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from xlearn.dataset import MNIST
from xlearn.nets.gans import WGAN
from xlearn.utils.image import subplot_images
from xlearn.utils.general import enter_debug, ProgressTimer


enter_debug()


def test(cfs):
    net = WGAN(filenames=cfs)
    net.define_net()



def train(nb_batches, cfs):

    GEN_FREQ = 2
    PRE_TRAIN = 100
    N_GEN_TRA = 0

    dataset = MNIST(filenames=cfs)
    print("=" * 30)
    print("DATASET SETTINGS:")
    print(dataset.pretty_settings())
    print("=" * 30)
    net = WGAN(filenames=cfs)
    net.define_net()
    print("=" * 30)
    print("NETWORK SETTINGS:")
    print(net.pretty_settings())
    print("=" * 30)
    ptp = ProgressTimer(PRE_TRAIN)
    for i in range(PRE_TRAIN):
        s = next(dataset)
        z = net.gen_latent()        
        loss_c = net.train_on_batch('Cri', [s[0], z], [])
        msg = 'loss_c= %f' % loss_c
        ptp.event(i, msg)
    # net._define_clip_steps()
    # _ = net.sess.run(net.clip_steps)
    pt = ProgressTimer(nb_batches)
    loss_c = np.nan
    loss_g = np.nan
    for i in range(nb_batches):
        s = next(dataset)
        z = net.gen_latent()
        if i % GEN_FREQ > 0:
            loss_c = net.train_on_batch('Cri', [s[0], z], [])
            # _ = net.sess.run(net.clip_steps)
            msg = 'c_step, loss_c= %f; loss_g= %f' % (loss_c, loss_g)
            pt.event(i, msg)
        else:
            loss_g = net.train_on_batch('WGan', [s[0], z], [])
            msg = 'g_step, loss_c= %f; loss_g= %f' % (loss_c, loss_g)
            pt.event(i, msg)
            N_GEN_TRA += 1
            if N_GEN_TRA > 25:
                GEN_FREQ = 2
            if N_GEN_TRA % 20 == 0:
                GEN_FREQ = 2
        if i % 1000 == 0:
            net.save('AutoEncoder')
    net.save('AutoEncoder', is_print=True)

def predict(cfs):
    dataset = MNIST(filenames=cfs)
    net = WGAN(filenames=cfs)
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
    net = WGAN(filenames=cfs)
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
        show_latent(config_files)
    elif task == 'test':
        test(config_files)
