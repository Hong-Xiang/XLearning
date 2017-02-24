import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from xlearn.utils.image import subplot_images
import h5py
from xlearn.dataset.sinogram import Sinograms
from xlearn.nets.super_resolution import SRNetInterp

from xlearn.utils.general import enter_debug
# enter_debug()


batch_size = 32
with Sinograms(file_data="/home/hongxwing/Workspace/Datas/shepplogan_sinograms.h5", is_gray=False, is_batch=True, batch_size=batch_size, is_down_sample=True, down_sample_ratio=[1, 4], is_padding=True, is_4d=True) as dataset:
    s = next(dataset)
    imgs = dataset.data_from_sample(s, data_type='data')        
    imgs = dataset.visualize(imgs)    
    plt.figure(figsize=(8, batch_size//8))
    subplot_images((imgs, ), is_gray=True)
    s = next(dataset)
    imgs = dataset.data_from_sample(s, data_type='label')        
    imgs = dataset.visualize(imgs)    
    plt.figure(figsize=(8, batch_size//8))
    subplot_images((imgs, ), is_gray=True)
net = SRNetInterp(shape_i=(365, 61, 1), shape_o=(365,244,1), down_sample_ratio=(1, 4, 1))
net.define_net()
with Sinograms(file_data="/home/hongxwing/Workspace/Datas/shepplogan_sinograms.h5", is_gray=False, is_batch=True, batch_size=batch_size, is_down_sample=True, down_sample_ratio=[1, 4], is_padding=True, is_4d=True) as dataset:
    s = next(dataset)
    imgs = dataset.data_from_sample(s, data_type='data')        
    imgs = dataset.visualize(imgs)    
    plt.figure(figsize=(8, batch_size//8))
    subplot_images((imgs, ), is_gray=True)
    p = net.predict(0, [s[0]])