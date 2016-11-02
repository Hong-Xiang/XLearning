from __future__ import absolute_import, division, print_function
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
import os.path
import supernet.input
from supernet.input import DataSet
from six.moves import xrange
import mypylib as mp
#check data
#DATA_DIR = "/home/hongxwing/Workspace/Datas/SinogramData/"
#PREFIX_HIGH = "sinogram_high_"
#PREFIX_LOW = "sinogram_low_"
PREFIX_PHANTOM = "phantom_"

DATA_DIR = "/home/hongxwing/Workspace/Datas/flickr25k_gray_npy/"
PREFIX_HIGH = "imh"
PREFIX_LOW = "iml"
SUFFIX = ".npy"

id = 0
fn_high = os.path.join(DATA_DIR, PREFIX_HIGH+str(id)+SUFFIX)
fn_low = os.path.join(DATA_DIR, PREFIX_LOW+str(id)+SUFFIX)
# fn_phantom = os.path.join(DATA_DIR, PREFIX_PHANTOM+str(id)+SUFFIX)

img_high = np.load(fn_high)
img_low = np.load(fn_low)
img_res = img_high - img_low
# img_phantom = np.load(fn_phantom)
plt.figure(figsize=[6,18])
plt.imshow(img_high,'gray')
plt.figure(figsize=[6,18])
plt.imshow(img_low,'gray')
plt.figure(figsize=[6,18])
plt.imshow(img_res,'gray')
plt.figure(figsize=[6,18])
# plt.imshow(img_phantom, 'gray', clim=[0.9,1.1])
# img_high.shape
#Check supernet.input: construct
# DATA_DIR = "/home/hongxwing/Workspace/Datas/SinogramData/"
# PREFIX_HIGH = "sinogram_high_"
# PREFIX_LOW = "sinogram_low_"
# PREFIX_PHANTOM = "phantom_"
# SUFFIX = ".npy"

n_files = 20000
#id_list = xrange(n_files)
id_list = [5951]
file_high = [os.path.join(DATA_DIR, PREFIX_HIGH + str(id) + SUFFIX) for id in id_list]
file_low = [os.path.join(DATA_DIR, PREFIX_LOW + str(id) + SUFFIX) for id in id_list]
data = DataSet(file_high, file_low,
                           [61, 61],
                           [8, 8],
                           use_random_shuffle=True,
                           max_patch_image=8)
for i in xrange(5000):
    print(i)
    data.next_batch(128)
