import tensorflow as tf
from xlearn.datasets.mnist_recon import MNISTRecon
import xlearn.nets as nets
import xlearn.datasets as datasets
from xlearn.utils.general import enter_debug


def main(*args, **kwargs):
    # enter_debug()
    tf.logging.set_verbosity(tf.logging.INFO)
    # net = MNISTRecon0(filenames='net_mnist_recon.json', load_step=-1)
    net = nets.Cali0(filenames='cali0.json')
    net.init()
    # with MNISTRecon(filenames='data_mnist_recon.json') as dataset:
    # with MNISTRecon(filenames='data_mnist_recon.json') as dataset:
    dataset_train = datasets.CalibrationDataSet(
        filenames='data_cali.json', mode='train')
    dataset_test = datasets.CalibrationDataSet(
        filenames='data_cali.json', mode='test')
    dataset_train.initialize()
    dataset_test.initialize()
    net.set_dataset({'train': dataset_train, 'test': dataset_test})
    net.train(steps=50000, phase=5, decay=10.0)
    dataset_train.finalize()
    dataset_test.finalize()
    net.save()


if __name__ == "__main__":
    tf.app.run()
