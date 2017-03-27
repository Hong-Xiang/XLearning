from keras.layers import Lambda
import keras.backend as K

def sub_fun(tensors):
    x, y = tensors
    return x - y

def sub(x, y):
    sub_layer = Lambda(sub_fun)
    return sub_layer([x, y])