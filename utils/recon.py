import numpy as np
import astra

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

def recon(sino, imgsize, num_sen, sen_width, theta):
    sino = sino.T
    vol_geom = astra.create_vol_geom(*imgsize)
    sino_geom = astra.create_proj_geom('parallel', sen_width, num_sen, theta)
    sinogram_id = astra.data2d.create('-sino', sino_geom, data=sino)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 10)
    rec = astra.data2d.get(rec_id)
    astra.data2d.clear()
    astra.projector.clear()
    astra.algorithm.clear()
    return rec

def padding_sino(sinos, crop_size=8, period=360):
    shape = list(sinos.shape)
    shape[1] += crop_size*2
    shape[2] += crop_size*2
    sino_full = np.zeros(shape)
    half_period = period // 2
    sino_full[:, crop_size:-crop_size, crop_size:-crop_size, :] = sinos
    for i in range(crop_size):        
        sino_full[:, :, i, :] = sino_full[:, ::-1, i+half_period, :]
    for i in range(crop_size):        
        sino_full[:, :, shape[2]-i-1, :] = sino_full[:, ::-1, shape[2]-i-1-half_period, :]
    return sino_full