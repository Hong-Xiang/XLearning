import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import h5py
from xlearn.utils.image import subplot_images
from xlearn.dataset.sinogram import Sinograms
from xlearn.dataset import Flickr25k, MNISTImage
from xlearn.nets.super_resolution import SRF3D
from xlearn.utils.tensor import downsample_shape
from xlearn.utils.general import enter_debug

# enter_debug()

patch_shape = (27, 27)
down_sample_ratio = (3, 3)
low_shape = downsample_shape(patch_shape, down_sample_ratio)
batch_size = 128
nb_batches = 1000
lr_now = 1e-4

dataset_name = 'mnist'

if dataset_name == 'sinograms':
    dataset_class = Sinograms
    dataset_path = "/home/hongxwing/Workspace/Datas/shepplogan_sinograms.h5"
elif dataset_name == 'flickr':
    dataset_class = Flickr25k
    dataset_path = "/home/hongxwing/Workspace/Datas/flickr25k.h5"
elif dataset_name == 'mnist':
    dataset_class = MNISTImage

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
    'lrs': [lr_now],
    'nb_deconv_filters': 64,
    'nb_input_filters': 64,
    'nb_input_row': 3,
    'nb_res_blocks': 2,
}


def train_3_phases(net, dataset):
    net.reset_lr([lr_now])
    for i in tqdm(range(nb_batches), ascii=True, ncols=50):
        s = next(dataset)
        net.train_on_batch(0, [s[0]], [s[1]])

    net.reset_lr([lr_now])
    for i in tqdm(range(nb_batches), ascii=True, ncols=50):
        s = next(dataset)
        net.train_on_batch(0, [s[0]], [s[1]])

    net.reset_lr([lr_now])
    for i in tqdm(range(nb_batches), ascii=True, ncols=50):
        s = next(dataset)
        net.train_on_batch(0, [s[0]], [s[1]])


def predict_and_show(net, dataset):
    s = next(dataset)
    p = net.predict(0, [s[0]])
    p = p[0]
    images = dataset.data_from_sample(s, data_type='data')
    images = dataset.visualize(images)
    labels = dataset.data_from_sample(s, data_type='label')
    labels = dataset.visualize(labels)    
    preds = dataset.visualize(p)
    subplot_images((images, labels, preds), is_gray=True)


def train():        
    net = SRF3D(**net_settings)
    print("net settings:")
    print(net.pretty_settings())
    net.define_net()

    if dataset_name == 'mnist':
        dataset = dataset_class(**data_settings)
        print("dataset settings:")
        print(dataset.pretty_settings())
        train_3_phases(net, dataset)
    else:
        with dataset_class(file_data=dataset_path, **data_settings) as dataset:
            print("dataset settings:")
            print(dataset.pretty_settings())
            train_3_phases(net, dataset)

    net.save(model_id=0)


def predict():
    net = SRF3D(**net_settings)
    print(net.pretty_settings())
    net.define_net()
    net.load(model_id=0)
    if dataset_name == 'mnist':
        dataset = dataset_class(data_settings)
        predict_and_show(net, dataset)
    else:
        with dataset_class(file_data=dataset_path, **data_settings) as dataset:
            predict_and_show(net, dataset)

if __name__ == "__main__":
    train()
