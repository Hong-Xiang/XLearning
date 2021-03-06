import tensorflow as tf
from xlearn.datasets.mnist_recon import MNISTRecon
import xlearn.nets as nets
import xlearn.datasets as datasets
from xlearn.utils.general import enter_debug
from xlearn.datasets.base import PATH_DATASETS
import sys
import h5py
import pathlib
import numpy as np
from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K

tf.flags.DEFINE_string('task', 'train', "running task")
tf.flags.DEFINE_integer('version', 0, "net_version")
flags = tf.flags.FLAGS


def trian(net):
    net.train(steps=[800000]*1, decay=10.0)
    net.save()


def predict(net, dataset):
    ss = dataset.sample()
    result = net.predict(ss)
    compare = [result['pos_pred'][:, 0], ss['label'][:, 0], result['pos_pred'][:, 1], ss['label'][:, 1]]
    compare = np.array(compare)
    print(compare)

def predict_cali(net):
    p = pathlib.Path(PATH_DATASETS) / 'cali.h5'
    with h5py.File(str(p.absolute()), 'r') as fin:
        data = np.array(fin['evt_all'])
        ipos = np.array(fin['inc_all'])
        ipos = ipos[:, :2]
        data = data.reshape([-1, 1, 10, 10])
        pred = net.predict_auto({'data': data})
        np.save('ipos.npy', ipos)
        np.save('pred.npy', pred)


def loss(pred, lable):
    error = pred - lable
    error = K.square(error)
    error = K.sum(error, axis=-1)
    error = K.sqrt(error + 1e-10)
    loss = K.mean(error)
    return loss

def main(*args, **kwargs):
    # enter_debug()
    tf.logging.set_verbosity(tf.logging.INFO)
    # net = MNISTRecon0(filenames='net_mnist_recon.json', load_step=-1)
    # with MNISTRecon(filenames='data_mnist_recon.json') as dataset:
    # with MNISTRecon(filenames='data_mnist_recon.json') as dataset:
    dataset_train = datasets.CalibrationDataSet(
        filenames='data_cali.json', mode='train')
    dataset_test = datasets.CalibrationDataSet(
        filenames='data_cali.json', mode='test')
    dataset_train.initialize()
    dataset_test.initialize()

    if flags.task == 'train':
        if flags.version == 0:
            net = nets.Cali0(filenames='cali0.json', load_step=-1)
        else:
            net = nets.Cali1(filenames='cali0.json', load_step=-1)
        net.init()
        net.set_dataset({'train': dataset_train, 'test': dataset_test})
        trian(net)
    elif flags.task == 'predict':
        if flags.version == 0:
            net = nets.Cali0(filenames='cali0.json', load_step=-1)
        else:
            net = nets.Cali1(filenames='cali0.json', load_step=-1)
        net.init()
        net.set_dataset({'train': dataset_train, 'test': dataset_test})
        predict(net, dataset_test)
    elif flags.task == 'predict_cali':
        if flags.version == 0:
            net = nets.Cali0(filenames='cali0.json', load_step=-1)
        else:
            net = nets.Cali1(filenames='cali0.json', load_step=-1)
        net.init()
        predict_cali(net)
    # elif flags.task == 'keras':

    #     ip = Input(shape=(1, 10, 10))
    #     h = Flatten()(ip)
    #     h = Dense(128, activation='relu')(h)
    #     h = Dense(256, activation='relu')(h)
    #     h = Dense(512, activation='relu')(h)
    #     h = Dense(1024, activation='relu')(h)
    #     h = Dense(2048, activation='relu')(h)
    #     h = Dense(4096, activation='relu')(h)
    #     h = Dense(8192, activation='relu')(h)
    #     h = Dropout(0.5)(h)
    #     out = Dense(2)(h)
    #     m = Model(ip, out)
    #     opt = RMSprop(1e-5)
    #     m.compile(loss=loss, optimizer=opt)
    #     m.summary()
    #     for i in range(1000):
    #         ss = dataset_train.sample()
    #         loss_v = m.train_on_batch(ss['data'], ss['label'])
    #         print(i, loss_v)
    dataset_train.finalize()
    dataset_test.finalize()


if __name__ == "__main__":
    tf.app.run(main, sys.argv)
