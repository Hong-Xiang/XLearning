from tqdm import tqdm

from xlearn.dataset import Flickr25k
from xlearn.utils.tensor import downsample_shape
from xlearn.utils.general import enter_debug

def test_infinite_loop():
    patch_shape = (64, 64)
    down_sample_ratio = (2, 2)
    low_shape = downsample_shape(patch_shape, down_sample_ratio)
    batch_size = 128
    nb_batches = 10000
    data_settings = {
        'is_batch': True,
        'batch_size': batch_size,
        'is_crop': True,
        'is_train': False,
        'crop_target_shape': patch_shape,
        'is_crop_random': True,
        'is_gray': True,
        'is_down_sample': True,
        'down_sample_ratio': down_sample_ratio,
        'down_sample_method': 'mean',
        'is_norm': True,
        'norm_c': 256.0,
    }

    with Flickr25k(**data_settings) as dataset:
        for _ in tqdm(range(nb_batches), ncols=100, ascii=True):
            s = next(dataset)

if __name__ == "__main__":
    enter_debug()
    test_infinite_loop()