import numpy as np


def reform(datas, transpose=False, flipx=False, flipy=False, per_rat=159):
    nb_rat = datas.shape[0]//per_rat
    out = np.zeros([nb_rat, per_rat, datas.shape[1], datas.shape[2]])
    cid = 0
    for i in range(nb_rat):
        for j in range(per_rat):
            tmp = datas[cid, :, :]
            if transpose:
                tmp = tmp.T
            if flipx:
                tmp = tmp[::-1, :]
            if flipy:
                tmp = tmp[:, ::-1]
            out[i, j] = tmp
            cid += 1
    return out


def clean_stir_recon(img, max_radius, min_radius=0.0, background_value=0.0):
    x_axis, x_step = np.linspace(-img.shape[0]//2, img.shape[0]//2, img.shape[0], False, True)
    y_axis, y_step = np.linspace(-img.shape[0]//2, img.shape[0]//2, img.shape[0], False, True)
    x_axis += x_step / 2
    y_axis += y_step / 2
    x, y = np.meshgrid(x_axis, y_axis)
    r = (x**2.0 + y**2.0)**(0.5)
    img[np.logical_and(r >= min_radius, r <= max_radius)] = background_value
    return img

