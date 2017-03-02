from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from xlearn.utils.image import subplot_images
from xlearn.utils.general import enter_debug
from xlearn.nets.gans import WGAN1D
from xlearn.dataset import MNIST

enter_debug()

batch_size = 16
nb_batches = 3000

data_settings = {
    'is_flatten': True,
    'is_unsp': False,
    'is_label': True,
    'is_weight': False,
    'is_batch': True,
    'is_bin': False,
    'is_norm': True,    
    'batch_size': batch_size,
    'is_cata': False,
}

net_settings = {
    'inputs_dims': ((28 * 28,), ),
    'latent_dims': (2,),
    'hiddens': (512, 512, 512),
    'batch_size': batch_size,
    'sigma': 1.0,
}


def test():
    net = WGAN1D(**net_settings)
    net.define_net()


def train():
    dataset = MNIST(**data_settings)
    net = WGAN1D(**net_settings)
    net.define_net()
    for i in tqdm(range(nb_batches)):
        z = net.gen_latent()
        s = next(dataset)
        for j in range(5):
            net.train_on_batch('Cri', [s[0], z])
        z = net.gen_latent()
        net.train_on_batch('Gen', [z])


def create_dataset_net(is_load=False):
    dataset = MNIST(**data_settings)
    net = WGAN1D(**net_settings)
    if is_load:
        net.load()
    return dataset, net


def predict(net, dataset, nb_batch=1):
    imgs = []
    for i in tqdm(range(nb_batch)):
        z = net.gen_latent()
        p = net.predict('Gen', [z])
        imgs += dataset.visualize(p[0])
    subplot_images((imgs, ), is_gray=True, size=3.0, tight_c=0.5)

if __name__ == "__main__":
    train()
