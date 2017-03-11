from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from xlearn.dataset import MNIST
from xlearn.nets.aae import AAE1D
from xlearn.utils.image import subplot_images
from xlearn.utils.general import enter_debug, ProgressTimer, with_config

enter_debug()


def test(cfs):
    print("Test Routine with config files:", cfs)
    net = AAE1D(filenames=cfs)
    net.define_net()


def train_ae(net, dataset):
    s = next(dataset)
    loss_ae = net.train_on_batch('ae', [s[0]], [], is_summary=False)
    return loss_ae


def train_cri(net, dataset):
    s = next(dataset)
    loss_cri = net.train_on_batch(
        'cri', [s[0], net.gen_latent()], [], is_summary=False)
    return loss_cri


def train_gen(net, dataset):
    s = next(dataset)
    loss_gen = net.train_on_batch('enc_logit', [s[0]], [], is_summary=False)
    return loss_gen


@with_config
def train(filenames=None, settings=None, **kwargs):
    dataset = MNIST(filenames=filenames)
    print("=" * 30)
    print("DATASET SETTINGS:")
    print(dataset.pretty_settings())
    print("=" * 30)
    net = AAE1D(filenames=filenames)
    net.define_net()
    print("=" * 30)
    print("NETWORK SETTINGS:")
    print(net.pretty_settings())
    print("=" * 30)
    nb_batches = settings['nb_batches']
    net.load()
    ptp = ProgressTimer(nb_batches)
    for i in range(nb_batches // 2):
        loss_ae = train_ae(net, dataset)
        msg = 'step #%5d, AuE, loss=%05f' % (net.step, loss_ae)
        ptp.event(net.step, msg)
    net.lr_decay()
    for i in range(nb_batches // 2):
        loss_ae = train_ae(net, dataset)
        msg = 'step #%5d, AuE, loss=%05f' % (net.step, loss_ae)
        ptp.event(net.step, msg)
    for i in range(nb_batches//5):
        loss_cri = train_cri(net, dataset)
        msg = 'Cri, loss=%05f    ' % (loss_cri)
        # loss_gen = train_gen(net, dataset)
        # msg += 'Gen, loss=%05f    ' % (loss_gen)
        ptp.event(net.step, msg)
        ptp = ProgressTimer(nb_batches)
    # for i in range(nb_batches // 2):
    #     loss_ae = train_ae(net, dataset)
    #     msg = 'step #%5d, AuE, loss=%05f' % (net.step, loss_ae)
    #     ptp.event(net.step, msg)
    # net.lr_decay()
    # for i in range(nb_batches // 2):
    #     loss_ae = train_ae(net, dataset)
    #     msg = 'step #%5d, AuE, loss=%05f' % (net.step, loss_ae)
    #     ptp.event(net.step, msg)
    for i in range(nb_batches):
        loss_cri = train_cri(net, dataset)
        msg = 'Cri, loss=%05f    ' % (loss_cri)
        loss_gen = train_gen(net, dataset)
        msg += 'Gen, loss=%05f    ' % (loss_gen)
        ptp.event(net.step, msg)
    net.save('net', is_print=True)
    print("PHASE " + "B" * 30)
    s = next(dataset)
    p = net.predict('cri', [s[0], net.gen_latent()])
    print('dataset::fake')
    print(p[0])
    print('mean', np.mean(p[0]))
    print('latent::true')
    print(p[1])
    print('mean', np.mean(p[1]))

#     for i in range(net.pre_train):
#         s = next(dataset)
#         z = net.gen_latent()
#         loss_c = net.train_on_batch('Cri', [s[0], z], [])
#         msg = 'loss_c= %f' % loss_c
#         ptp.event(i, msg)
#     pt = ProgressTimer(nb_batches)
#     loss_c = np.nan
#     loss_g = np.nan
#     for i in range(nb_batches):
#         s = next(dataset)
#         z = net.gen_latent()
#         if i % net.gen_freq > 0:
#             loss_c = net.train_on_batch('Cri', [s[0], z], [])
#             msg = 'c_step, loss_c= %f; loss_g= %f' % (loss_c, loss_g)
#             pt.event(i, msg)
#         else:
#             loss_g = net.train_on_batch('WGan', [s[0], z], [])
#             msg = 'g_step, loss_c= %f; loss_g= %f' % (loss_c, loss_g)
#             pt.event(i, msg)
#         if i % 1000 == 0:
#             net.save('AutoEncoder')
#     net.save('AutoEncoder', is_print=True)


def show_autoencoder(cfs):
    dataset = MNIST(filenames=cfs)
    net = AAE1D(filenames=cfs)
    net.define_net()
    print(net.pretty_settings())
    net.load('AutoEncoder')
    x = net.gen_latent()
    p = net.predict('ae', [x])
    imgs = dataset.visualize(p[0])
    subplot_images((imgs, ), is_gray=True, size=3.0, tight_c=0.5)


def show_latent(cfs, nb_sample=10000):
    print("AE1D Test. Show latent called.")
    net = AAE1D(filenames=cfs)
    net.define_net()
    net.load('enc')
    nb_batch_show = nb_sample // net._batch_size
    dataset = MNIST(filenames=cfs)
    latents = []
    for i in range(10):
        latents.append([])
    for i in tqdm(range(nb_batch_show)):
        s = next(dataset)
        p = net.predict('enc', [s[0]])
        for j in range(dataset._batch_size):
            latents[s[1][j]].append(p[0][j])
    x = []
    y = []
    for i in range(10):
        pos = np.array(latents[i])
        x.append(pos[:, 0])
        y.append(pos[:, 1])
    para = []
    for i in range(10):
        para.append(x[i])
        para.append(y[i])
        para.append('.')
    plt.plot(*para)
    plt.legend(list(map(str, range(10))))


def show_data_mainfold(cfs):
    """ show latent main fold for data """
    dataset = MNIST(filenames=cfs)
    nb_axis = int(np.sqrt(dataset._batch_size))
    x = np.linspace(-1.5, 1.5, nb_axis)
    y = np.linspace(-1.5, 1.5, nb_axis)
    pos = np.meshgrid(x, y)
    xs = pos[0]
    ys = pos[1]
    xs = xs.reshape([-1])
    ys = ys.reshape([-1])
    net = AAE1D(filenames=cfs)
    net.define_net()
    net.load('Gen')
    latents = np.array([xs, ys]).T
    pall = None
    nb_latents = latents.shape[0]
    nb_batches = int(np.ceil(nb_latents / net.batch_size))
    nb_pad = nb_batches * net.batch_size - nb_latents
    latents_pad = np.pad(latents, ((0, nb_pad), (0, 0)), mode='constant')
    for i in tqdm(range(nb_batches)):
        data_batch = latents_pad[
            i * net.batch_size:(i + 1) * net.batch_size, :]
        p = net.predict('Gen', [data_batch])
        if pall is None:
            pall = p[0]
        else:
            pall = np.concatenate((pall, p[0]))
    p = pall[:nb_latents, ...]
    imgs = dataset.visualize(p)
    subplot_images((imgs, ), nb_max_row=nb_axis,
                   is_gray=True, size=1.0, tight_c=0.5)

if __name__ == "__main__":
    task = sys.argv[1]
    config_files = sys.argv[2:]
    print("Running with task:", task)
    if task == 'train':
        train(filenames=config_files)
    elif task == 'latent':
        show_latent(config_files)
    elif task == 'show_autoencoder':
        show_autoencoder(config_files)
    elif task == 'data':
        show_data_mainfold(config_files)
    elif task == 'test':
        test(config_files)
