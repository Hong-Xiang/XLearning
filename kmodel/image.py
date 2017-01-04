from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Convolution2D, BatchNormalization, Merge, ELU, LeakyReLU, UpSampling2D


def residual_block(input_, channels, kxs, kys, id=0, activation='elu'):
    n_conv = len(channels)
    x = input_
    for i in range(n_conv):
        x = Convolution2D(channels[i], kxs[i], kys[
                          i], border_mode='same', name='sr_res_conv_%d_%d' % (id, i))(x)
        x = BatchNormalization(name='sr_res_bn_%d_%d' % (id, i))(x)
        x = LeakyReLU(alpha=0.25, name="sr_res_activation_%d_%d" % (id, i))(x)
    m = Merge([x, input_], mode='sum', name="sr_res_merge_%d" % id)
    return m

def upscale_block(input_, r, id):
    x = input_
    x = Convolution2D(128, 3, 3, border_mode='same', name='sr_res_upconv1_%d' % id)(x)
    x = LeakyReLU(alpha=0.25, name='sr_res_up_lr_%d_1_1' % id)(x)
    x = UpSampling2D(size=(r, r), name='sr_res_upscale_%d' % id)(x)
    #x = SubPixelUpscaling(r=2, channels=32)(x)
    x = Convolution2D(128, 3, 3, border_mode='same', name='sr_res_filter1_%d' % id)(x)
    x = LeakyReLU(alpha=0.3, name='sr_res_up_lr_%d_1_2' % id)(x)
    return x

def conv_seq(input_, channels, kxs, kys, id=0, activation='elu'):
    n_conv = len(channels)
    x = input_
    for i in range(n_conv):
        x = Convolution2D(channels[i], kxs[i], kys[
                          i], border_mode='same', name='seq_conv_conv_%d_%d' % (id, i))(x)
        x = BatchNormalization(name='seq_conv_bn_%d_%d' % (id, i))(x)
        x = LeakyReLU(alpha=0.25, name="seq_conv_activation_%d_%d" % (id, i))(x)    
    return x
