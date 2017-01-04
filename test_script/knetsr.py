import sys
import numpy as np
from xlearn.reader.srinput import DataSetSuperResolution
from xlearn.knet.sr import KNetSR

def main(argv):
    """ Train Fx Net """
    dataset = DataSetSuperResolution(['./super_resolution.json', './sino_train.json'])    
    net = KNetSR(filenames=['./super_resolution.json', './netsr.json', './sino_train.json'])
    net.define_net()
    # net.model.fit(x, y, nb_epoch=10, batch_size=1024)
    net.model.fit_generator(dataset, samples_per_epoch=100, nb_epoch=10, callbacks=net.callbacks)
    x, y = dataset.next_batch()
    p = net.model.predict(x)
    np.save(x,'x.npy')
    np.save(y,'y.npy')
    np.save(p,'p.npy')

if __name__ == "__main__":
    main(sys.argv)
