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
