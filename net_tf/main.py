import tensorflow as tf

from xlearn.datasets.mnist_recon import MNISTRecon
import xlearn.model_fns.mnist_recon as mr

def main(*args, **kwargs):
    with MNISTRecon(filenames='mnist_recon.json') as dataset:
        nn = tf.estimator.Estimator(mr.mnist_recon, './model')
        nn.train(dataset.sample, steps=100)

if __name__ == "__main__":
    tf.app.run()