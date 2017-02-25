import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import h5py
from xlearn.utils.image import subplot_images
from xlearn.dataset.sinogram import Sinograms
from xlearn.dataset import Flickr25k
from xlearn.nets.super_resolution import SRF3D
from xlearn.utils.tensor import downsample_shape
from xlearn.utils.general import enter_debug

# enter_debug()

patch_shape = (120, 120)
down_sample_ratio = (3, 3)
low_shape = downsample_shape(patch_shape, down_sample_ratio)
batch_size = 64
nb_batches = 1000

# dataset_class = Sinograms
# dataset_path = "/home/hongxwing/Workspace/Datas/shepplogan_sinograms.h5"
dataset_class = Flickr25k
dataset_path = "/home/hongxwing/Workspace/Datas/flickr25k.h5"

data_settings = {
    'is_batch': True,
    'batch_size': batch_size,
    'is_crop': True,
    'crop_target_shape': patch_shape,
    'is_crop_random': True,
    'is_gray': True,
    'is_down_sample': True,
    'down_sample_ratio': down_sample_ratio,
    'down_sample_method': 'mean',
    'is_norm': True,
    'norm_c': 256.0,
}

net_settings = {
    'shape_i': tuple(list(low_shape) + [1]),
    'shape_o': tuple(list(patch_shape) + [1]),
    'down_sample_ratio': tuple(list(down_sample_ratio) + [1]),
    'batch_size': batch_size,
    'lrs': [1e-4]
}

net = SRF3D(**net_settings)
print(net.pretty_settings())
net.define_net()


with dataset_class(file_data=dataset_path, **data_settings) as dataset:
    net.reset_lr([1e-4])
    for i in tqdm(range(nb_batches), ascii=True, ncols=50):
        s = next(dataset)
        net.train_on_batch(0, [s[0]], [s[1]])

    net.reset_lr([1e-5])
    for i in tqdm(range(nb_batches), ascii=True, ncols=50):
        s = next(dataset)
        net.train_on_batch(0, [s[0]], [s[1]])

    net.reset_lr([1e-6])
    for i in tqdm(range(nb_batches), ascii=True, ncols=50):
        s = next(dataset)
        net.train_on_batch(0, [s[0]], [s[1]])

net.save(model_id=0)
