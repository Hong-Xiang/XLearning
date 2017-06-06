import numpy as np
import astra
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

def rmse(label, target):
    err = target - label
    value = np.sqrt(np.sum(np.square(err))/label.size)
    return value

def psnr(label, target):
    minv = np.min(label)
    maxv = np.max(label)
    sca = 255.0 / (maxv - minv)
    ln = (label - minv) * sca
    tn = (target - minv) * sca
    rmv = rmse(ln, tn)
    value = 10 * np.log((255.0**2)/(rmv**2))/np.log(10)
    return value




'''
The function to compute SSIM
@param param: img_mat_1 1st 2D matrix
@param param: img_mat_2 2nd 2D matrix
'''
def ssim(label, target):
    minv = np.min(label)
    maxv = np.max(label)
    sca = 255.0 / (maxv - minv)
    img_mat_1 = (label - minv) * sca
    img_mat_2 = (target - minv) * sca
    import numpy
    import scipy.ndimage
    from numpy.ma.core import exp
    from scipy.constants.constants import pi
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=numpy.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(numpy.float)
    img_mat_2=img_mat_2.astype(numpy.float)
    
    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)
        
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)
    
    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;
    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2
    
    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=numpy.average(ssim_map)

    return index


def proj(img, num_sen, sen_width, theta):    
    shape = img.shape
    if not len(shape) == 2:
        raise ValueError('Invalid img shape {}.'.format(shape))
    vol_geom = astra.create_vol_geom(*shape)
    proj_geom = astra.create_proj_geom('parallel', sen_width, num_sen, theta)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    sinogram_id, sinogram = astra.create_sino(img, proj_id)
    sinogram = np.array(sinogram)
    astra.data2d.clear()
    astra.projector.clear()
    astra.algorithm.clear()
    return sinogram

def recon(sino, imgsize, num_sen, sen_width, theta, method='FBP_CUDA', run_time=1):
    sino = np.array(sino)
    if len(sino.shape) == 4:
        nb_sample = sino.shape[0]
        recs = []
        for i in tqdm_notebook(range(nb_sample)):
            recs.append(recon(sino[i, :, :, 0], imgsize, num_sen, sen_width, theta))
        rec = np.array(recs)
        rec = np.reshape(rec, list(rec.shape)+[1])
        return rec
    if not len(sino.shape) == 2:
        raise ValueError('Invalid shape of sinogram.')
    sino = sino.T
    sino_shape = sino.shape
    if not sino_shape[1] == num_sen:
        raise ValueError('Invalid num_sen, sino_shape: {}, num_sen: {}.'.format(sino_shape, num_sen))
    if not sino_shape[0] == len(theta):
        raise ValueError('Invalid angles, sino_shape: {}, len(theta): {}.'.format(sino_shape, len(theta)))
    vol_geom = astra.create_vol_geom(*imgsize)
    sino_geom = astra.create_proj_geom('parallel', sen_width, num_sen, theta)
    sinogram_id = astra.data2d.create('-sino', sino_geom, data=sino)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, run_time)
    rec = astra.data2d.get(rec_id)
    astra.data2d.clear()
    astra.projector.clear()
    astra.algorithm.clear()
    return rec

def padding_sino(sinos, crop_size=8, period=360, mid=False):
    shape = list(sinos.shape)
    if not isinstance(crop_size, (list, tuple)):
        crop_size = [crop_size, crop_size]
    shape[1] += crop_size[0]*2
    shape[2] += crop_size[1]*2
    sino_full = np.zeros(shape)
    half_period = period // 2
    crop_size = crop_size[0]
    if not mid:
        sino_full[:, crop_size:-crop_size, crop_size:-crop_size, :] = sinos
        for i in range(crop_size):        
            sino_full[:, :, i, :] = sino_full[:, ::-1, i+half_period, :]
        for i in range(crop_size):        
            sino_full[:, :, shape[2]-i-1, :] = sino_full[:, ::-1, shape[2]-i-1-half_period, :]
    else:
        sino_full[:, crop_size+crop_size//2:-crop_size//2, crop_size+crop_size//2:-crop_size//2, :] = sinos
        for i in range(crop_size+crop_size//2):        
            sino_full[:, :, i, :] = sino_full[:, ::-1, i+half_period, :]        
        for i in range(crop_size//2):        
            sino_full[:, :, shape[2]-i-1, :] = sino_full[:, ::-1, shape[2]-i-1-half_period, :]
    return sino_full

def denorm(datas, mean_values, std_values):
    nb_samples = datas.shape[0]
    datas = np.array(datas)
    for i in range(nb_samples):
        datas[i] *= std_values[i]
        datas[i] += mean_values[i]
    return datas

def process(data0, data1, data2, data3, sino_sr, sino_it, phan, num_sen, sen_width, theta, half_recon=False, crop_size=8, period=360):
    ida = list(range(64,64+180))
    theta0 = theta[:]
    theta1 = theta[1::2]
    theta2 = theta[2::4]
    theta3 = theta[4::8]
    shape = phan.shape
    rec0 = recon(data0[:, :, ida, :], shape[1:3], num_sen, sen_width, theta0[ida])
    rec1 = recon(data1, shape[1:3], num_sen//2, sen_width*2, theta1)/2
    rec2 = recon(data2, shape[1:3], num_sen//4, sen_width*4, theta2)/4
    rec3 = recon(data3, shape[1:3], num_sen//8, sen_width*8, theta3)/8
    rec_sr = recon(sino_sr[:, :, ida, :], shape[1:3], num_sen, sen_width, theta0[ida])
    rec_it = recon(sino_it[:, :, ida, :], shape[1:3], num_sen, sen_width, theta0[ida])
    esino_sr = sino_sr - data0
    esino_it = sino_it - data0
    esino_sr_v = np.sum(np.sum(np.sum(np.square(esino_sr), axis=1), axis=1), 1)/(np.prod(sino_sr.shape[1:]))
    esino_it_v = np.sum(np.sum(np.sum(np.square(esino_it), axis=1), axis=1), 1)/(np.prod(sino_sr.shape[1:]))
    return rec0, rec1, rec2, rec3, rec_sr, rec_it, esino_sr, esino_it, esino_sr_v, esino_it_v

def plot_result(sino0, sino1, sino2, sino3, ssr, sit, img0, img1, img2, img3, imgsr, imgit, phan, idxs=(0,)):
    for i in idxs:
        sino0s = sino0[i, :, :, 0]
        sino1s = sino1[i, :, :, 0]
        sino2s = sino2[i, :, :, 0]
        sino3s = sino3[i, :, :, 0]
        img0s = img0[i, :, :, 0]
        img1s = img1[i, :, :, 0]
        img2s = img2[i, :, :, 0]
        img3s = img3[i, :, :, 0]
        imgsrs = imgsr[i, :, :, 0]
        imgits = imgit[i, :, :, 0]
        phans = phan[i, :, :, 0]
        ssrs = ssr[i, :, :, 0]
        sits = sit[i, :, :, 0]
        plt.figure(figsize=(32, 32*4))
        plt.gray()
        plt.subplot(1, 4, 1)
        plt.imshow(sino0s)
        plt.subplot(1, 4, 2)
        plt.imshow(sino1s)
        plt.subplot(1, 4, 3)
        plt.imshow(sino2s)
        plt.subplot(1, 4, 4)
        plt.imshow(sino3s)
        sinos = np.concatenate([sino0s, ssrs, sits], axis=1)
        plt.figure(figsize=(32, 32*3))
        plt.imshow(sinos)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        imgss = np.concatenate([img0s, img1s, img2s, img3s], axis=1)
        plt.figure(figsize=(32, 32*4))
        plt.imshow(imgss)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        imgrs = np.concatenate([phans, img0s, imgsrs, img2s, imgits], axis=1)
        plt.figure(figsize=(32, 32*5))
        plt.imshow(imgrs)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

