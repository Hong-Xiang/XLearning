import sys
import numpy as np
import xlearn.reader.fx
from xlearn.knet.ca import KNetCa


def main(argv):
    """ Train Ca Net """
    data = np.load(
        '/home/hongxwing/Workspace/Datas/DetectorCalibration/datas/data.npy')
    label = np.load(
        '/home/hongxwing/Workspace/Datas/DetectorCalibration/datas/label.npy')
    label = label[:, 0, 0]
    net = KNetCa(filenames=['./knetca.json'])
    net.define_net()
    net.model.fit(data, label, nb_epoch=100,
                  batch_size=128, callbacks=net.callbacks)
    # net.model.fit_generator(data_gen, nb_epoch=10, batch_size=1024, callbacks=net.callbacks)

if __name__ == "__main__":
    main(sys.argv)
