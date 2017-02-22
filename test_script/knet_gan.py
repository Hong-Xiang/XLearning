""" Training GAN net on MNIST dataset """
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from ..dataset.mnist import MNIST


def plot_gen(net, n_ex=16, dim=(4, 4), figsize=(10, 10), encoding_dim=32):
    noise = np.random.uniform(size=[n_ex, encoding_dim])
    generated_images = net.model_gen.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, :, :, 0]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_loss(losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 4))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()


def pretrain_dis(net, dataset, n_batch=10):
    for _ in tqdm(range(n_batch), ascii=True):
        s = next(dataset)
        x = s[0]    
        s_full = net.prepare_data(x)
        # net.model_dis.fit(s_full[0], s_full[1], nb_epoch=1)
        net.train_on_batch_n(model_id=1, inputs=s_full[0], outputs=s_full[1])
    losses = net.loss_records[1]
    x = list(range(len(losses)))
    # plt.plot(x, losses)

def train_gan(net, dataset, nb_batch=10, batch_size=32, encoding_dim=128, plot_freq=10):
    losses = {"d": [], "g": []}
    for i in tqdm(range(nb_batch), ascii=True, ncols=100):
        s = next(dataset)
        x = s[0]
        s_full = net.prepare_data(x)
        ld = net.model_dis.train_on_batch(s_full[0], s_full[1])
        losses['d'].append(ld)

        l = net.gen_noise()
        batch_size = l.shape[0]
        y = np.zeros([batch_size, 2])
        y[:, 0] = 1
        lg = net.model_GAN.train_on_batch(l, y)
        losses['g'].append(lg)
        if i % plot_freq == 0:
            plot_loss(losses)
            plot_gen(net, encoding_dim=encoding_dim)


def main():
    pass

if __name__ == "__main__":
    main()
