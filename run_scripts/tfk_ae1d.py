from xlearn.dataset import MNIST
from xlearn.nets.autoencoders import AutoEncoder1D
from tqdm import tqdm

batch_size = 32
nb_batches = 1000

data_settings = {
    'is_flatten': True,
    'is_unsp': True,
    'is_label': False,
    'is_weight': False,
    'is_batch': True,
    'is_bin': True,
    'is_norm': True,
    'batch_size': batch_size
}

net_settings = {
    'inputs_dims': ((28 * 28,), ),
    'hiddens': (128, 28 * 28)
}


def train():
    dataset = MNIST(**data_settings)
    net = AutoEncoder1D(**net_settings)
    net.define_net()
    for i in tqdm(range(nb_batches), ascii=True, ncols=50):
        s = next(dataset)
        print(' loss = ', net.train_on_batch(0, [s[0]], [s[0]]))
    net.save(model_id=0)

if __name__ == "__main__":
    train()
