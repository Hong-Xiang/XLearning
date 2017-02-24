import numpy as np
from tqdm import tqdm
from ..knet.aemnist import WGAN
from ..dataset.mnist import MNIST


def train_cri(net, dataset, n_batch=10):
    for i in range(n_batch):
        s = next(dataset)
        t_img = s[0]
        n_img = net.gen_data()
        t1 = np.ones((net.batch_size, 1))
        t2 = - np.ones((net.batch_size, 1))
        c_input = np.concatenate((t_img, n_img))
        c_output = np.concatenate((t1, t2))
        net.clip_theta()
        net.train_on_batch('cri', c_input, c_output)        




def train_wgan(net, dataset, n_batch=1):
    for i in range(n_batch):
        noise_input = net.gen_noise()
        t1 = np.ones((net.batch_size, 1))
        net.cri.trainable = False
        net.train_on_batch('wgan', noise_input, t1)
        net.cri.trainable = True

