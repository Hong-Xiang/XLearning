from ..knet.aemnist import AutoEncoder
from ..dataset.mnist import MNIST


def test_AutoEncoder():
    BATCH_SZ = 64
    dataset_train = MNIST(is_noise=False, noise_scale=1, noise_type='poisson',
                          is_unsp=True, is_flatten=True, is_norm=True, is_batch=True, batch_size=BATCH_SZ)
    dataset_test = MNIST(is_train=False, is_noise=False, noise_scale=1, noise_type='poisson',
                         is_unsp=True, is_flatten=True, is_norm=True, is_batch=True, batch_size=32)
    s = next(dataset_train)
    imgs_d = dataset_train.visualize(s, image_type='data')
    imgs_l = dataset_train.visualize(s, image_type='label')
    plt.figure(figsize=(8, 8))
    uti.subplot_images((imgs_d[:32], imgs_l[:32]), is_gray=True)
    net = AutoEncoder(lr_init=1e-3)
    net.define_net()
    net.model[0].fit_generator(dataset_train, samples_per_epoch=60000,
                               nb_epoch=10, validation_data=dataset_test, nb_val_samples=1024)
    test_sample = next(dataset_test)
    predict = net.autoencoder.predict(test_sample[0])
    formated_sample = (predict, test_sample[1], 1.0)
    img_d = dataset_test.visualize(test_sample, image_type='data')
    img_l = dataset_test.visualize(test_sample, image_type='label')
    img_p = dataset_test.visualize(formated_sample, image_type='data')
    plt.figure(figsize=(12, 8))
    uti.subplot_images((img_d[:32], img_l[:32], img_p[:32]), is_gray=True)
