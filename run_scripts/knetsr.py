import sys
import numpy as np
from xlearn.reader.srinput import DataSetSuperResolution
from xlearn.knet.sr import KNetSR


def main(argv):
    """ Train Fx Net """
    dataset = DataSetSuperResolution(
        ['./super_resolution.json', './sino_train.json'])
    dataset_test = DataSetSuperResolution(
        ['./super_resolution.json', './sino_test.json'])
    net = KNetSR(filenames=['./super_resolution.json',
                            './netsr.json', './sino_train.json'])
    net.define_net()
    # net.model.fit(x, y, nb_epoch=10, batch_size=1024)
    net.model.fit_generator(dataset, samples_per_epoch=1024,
                            nb_epoch=10, callbacks=net.callbacks)
    net.model.evaluate_generator(dataset_test, val_samples=100)
    x, y = dataset_test.next_batch()
    p = net.model.predict(x)
    np.save('x.npy', x)
    np.save('y.npy', y)
    np.save('p.npy', p)

if __name__ == "__main__":
    main(sys.argv)
