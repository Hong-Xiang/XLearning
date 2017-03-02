from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from xlearn.dataset import MNIST
from xlearn.nets.autoencoders import AutoEncoder1D, VAE1D
from xlearn.utils.image import subplot_images
from xlearn.utils.general import enter_debug


enter_debug()


net_name = 'vae'

batch_size = 16
# nb_batches = 60000*75//batch_size+1
nb_batches = 30000

data_settings = {
    'is_flatten': True,
    'is_unsp': False,
    'is_label': True,
    'is_weight': False,
    'is_batch': True,
    'is_bin': False,
    'is_norm': True,
    'lrs': [1e-3],
    'batch_size': batch_size,
    'is_cata': False,
}

net_settings = {
    'inputs_dims': ((28 * 28,), ),
    'latent_dims': 2,
    'hiddens': (512, 512, 512),
    'batch_size': batch_size,
    'sigma': 1.0,
}


def test():
    net = VAE1D(**net_settings)


def train(is_print_loss=False):
    dataset = MNIST(**data_settings)
    if net_name == 'ae':
        net = AutoEncoder1D(**net_settings)
    elif net_name == 'vae':
        net = VAE1D(**net_settings)

    net.define_net()
    # net.load()
    print(net.pretty_settings())
    for i in tqdm(range(nb_batches), ascii=True, ncols=50):
        s = next(dataset)
        loss_v = net.train_on_batch('AutoEncoder', [s[0]], [s[0]])
        if is_print_loss:
            print(' loss = ', loss_v)
        if i % 1000 == 0:
            net.save('AutoEncoder')
    # for i in tqdm(range(nb_batches), ascii=True, ncols=50):
    #     net.reset_lr([1e-3])
    #     s = next(dataset)
    #     loss_v = net.train_on_batch('AutoEncoder', [s[0]], [s[0]])
    #     if is_print_loss:
    #         print(' loss = ', loss_v)
    #     if i % 1000 == 0:
    #             net.save('AutoEncoder')
    # for i in tqdm(range(nb_batches), ascii=True, ncols=50):
    #     net.reset_lr([1e-3])
    #     s = next(dataset)
    #     loss_v = net.train_on_batch('AutoEncoder', [s[0]], [s[0]])
    #     if is_print_loss:
    #         print(' loss = ', loss_v)
    #     if i % 1000 == 0:
    #         net.save('AutoEncoder')
    net.save('AutoEncoder', is_print=True)


def predict():
    dataset = MNIST(**data_settings)
    if net_name == 'ae':
        net = AutoEncoder1D(**net_settings)
    elif net_name == 'vae':
        net = VAE1D(**net_settings)
    net.define_net()
    print(net.pretty_settings())
    net.load('AutoEncoder')
    s = next(dataset)
    p = net.predict('Decoder', [net.gen_latent()])
    imgs = dataset.visualize(p[0])
    subplot_images((imgs, ), is_gray=True, size=3.0, tight_c=0.5)

def show_latent(nb_sample=10000):
    print("AE1D Test. Show latent called.")
    if net_name == 'ae':
        net = AutoEncoder1D(**net_settings)
    elif net_name == 'vae':
        net = VAE1D(**net_settings)
    net.define_net()    
    net.load('AutoEncoder')    
    nb_batch_show = nb_sample // batch_size
    dataset = MNIST(**data_settings)
    latents = []
    for i in range(10):
        latents.append([])
    for i in tqdm(range(nb_batch_show)):
        s = next(dataset)
        p = net.predict('Encoder', [s[0]])
        for j in range(batch_size):
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

def show_mainfold():
    dataset = MNIST(**data_settings)
    nb_axis = int(np.sqrt(batch_size))
    x = np.linspace(-1.5, 1.5, nb_axis)
    y = np.linspace(-1.5, 1.5, nb_axis)
    pos = np.meshgrid(x, y)
    xs = pos[0]
    ys = pos[1]
    xs = xs.reshape([-1])
    ys = ys.reshape([-1])
    if net_name == 'ae':
        net = AutoEncoder1D(**net_settings)
    elif net_name == 'vae':
        net = VAE1D(**net_settings)
    net.define_net()    
    net.load('AutoEncoder')
    latents = np.array([xs, ys]).T    
    p = net.predict('Decoder', [latents])
    imgs = dataset.visualize(p[0])
    subplot_images((imgs, ), nb_max_row=nb_axis, is_gray=True, size=1.0, tight_c=0.5)

if __name__ == "__main__":    
    if len(sys.argv) == 1:
        task = 'train'
    else:
        task = sys.argv[1]
    print("Running with task:", task)
    if task == 'train':
        if len(sys.argv) > 2:
            net_settings['path_summary'] = [sys.argv[2]]
            net_settings['sigma'] = float(sys.argv[3])            
        train()
    elif task == 'predict':
        predict()    
    elif task == 'show':
        show_latent()
    elif task == 'test':
        test()
