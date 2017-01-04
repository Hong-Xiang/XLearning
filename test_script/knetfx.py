import sys
import numpy as np
import xlearn.reader.fx
from xlearn.knet.fx import KNetFx

def main(argv):
    """ Train Fx Net """
    dataset = xlearn.reader.fx.DataSetFx(
        filenames="./fx.json", batch_size=1024 * 128, is_gaussian=True)
    x, y = dataset.next_batch()
    net = KNetFx(filenames=['./fx.json', './net.json'])
    net.define_net()
    data_gen = dataset
    net.model.fit(x, y, nb_epoch=10, batch_size=1024, callbacks=net.callbacks)
    # net.model.fit_generator(data_gen, nb_epoch=10, batch_size=1024, callbacks=net.callbacks)
    x_test = np.linspace(-1, 1, 256)
    y_pred = net.model.predict(x_test)
    if len(argv) == 1:
        fn_x = 'x.npy'
        fn_y = 'y.npy'
    else:
        fn_x = argv[1]
        fn_y = argv[2]
    np.save(fn_x, x_test)
    np.save(fn_y, y_pred)

if __name__ == "__main__":
    main(sys.argv)
