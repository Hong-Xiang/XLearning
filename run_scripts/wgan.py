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
    dataset = MNIST(filenames=cfs)
    net = WGAN(filenames=cfs)
    net.define_net()
    print(net.pretty_settings())
    pt = ProgressTimer(nb_batches)
    for i in range(nb_batches):
        s = next(dataset)
        loss_v = net.train_on_batch('AutoEncoder', [s[0]], [s[0]])
        msg = 'loss= %f' % loss_v
        pt.event(i, msg)
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


# def show_latent(cfs, nb_sample=10000):
#     print("WGAN Test. Show latent called.")
#     net = VAE1D(filenames=cfs)
#     net.define_net()
#     net.load('AutoEncoder')
#     nb_batch_show = nb_sample // net._batch_size
#     dataset = MNIST(filenames=cfs)
#     latents = []
#     for i in range(10):
#         latents.append([])
#     for i in tqdm(range(nb_batch_show)):
#         s = next(dataset)
#         p = net.predict('Encoder', [s[0]])
#         for j in range(dataset._batch_size):
#             latents[s[1][j]].append(p[0][j])
#     x = []
#     y = []
#     for i in range(10):
#         pos = np.array(latents[i])
#         x.append(pos[:, 0])
#         y.append(pos[:, 1])
#     para = []
#     for i in range(10):
#         para.append(x[i])
#         para.append(y[i])
#         para.append('.')
#     plt.plot(*para)
#     plt.legend(list(map(str, range(10))))


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
    net = VAE1D(filenames=cfs)
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
