from keras.layers import Conv2D, Dense, Flatten, ELU, Input, concatenate, Dropout, MaxPool2D, add, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
import numpy as np
import sys
import random
import os


def data_aug(x, y):
    x2 = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x2[i, :, :, 0] = x[i, :, :, 0].T
    x = np.concatenate([x, x])
    y = np.concatenate([y, y])
    return x, y


def load_data():
    datapath = '/home/hongxwing/Workspace/cali/data/transfer/data_grid2_100/'
    valid_label = np.load(os.path.join(datapath, 'valid_label.npy'))
    opms = np.load(os.path.join(datapath, 'opms.npy'))
    print(valid_label.shape)
    print(opms.shape)    
    data = np.array(opms)    
    # sum_data = np.sum(data, axis=1)
    # sum_data = np.sum(sum_data, axis=1)
    # idx = np.nonzero(sum_data > 5500)[0]
    # data = data[idx, :, :]
    # valid_label = valid_label[idx]
    label = np.array(valid_label)
    label = to_categorical(label, 2)
    nb_data = data.shape[0]
    idx = list(range(nb_data))
    data = data[idx, :, :]
    label = label[idx, :]
    nb_train = nb_data // 5 * 4
    data = np.reshape(data, [nb_data, 10, 10, 1])
    train_x = data[:nb_train, :, :, :]
    test_x = data[nb_train:, :, :, :]
    train_y = label[:nb_train, :]
    test_y = label[nb_train:, :]
    return (train_x, train_y), (test_x, test_y)


def train(m, train_data, test_data):
    # enter_debug()
    nb_epochs = 1
    for i in range(100):
        print("TRAING %d" % i)
        # m.fit(list(train_data[0]), train_data[1], batch_size=128,
        #       epochs=nb_epochs, validation_data=test_data)
        print(train_data[0].shape)
        print(train_data[1].shape)
        m.fit(train_data[0], train_data[1], batch_size=2048,
              epochs=nb_epochs, validation_data=test_data)
        m.save('save-%d' % i)
        # pdx = m.predict(test_data[0])
        # lbx = test_data[1]
        # err = lbx - pdx
        # err_s = np.sqrt(np.mean(np.square(err)))
        # err_s = np.mean(np.abs(err))
        # print('err_test', err_s)


# def model_define():
#     ip = Input(shape=(10, 10, 1))
#     h = Flatten()(ip)
#     h = Dense(128, activation='relu')(h)
#     h = Dense(512, activation='relu')(h)
#     h = Dropout(0.5)(h)
#     out = Dense(2, activation='sigmoid')(h)
#     m = Model(ip, out)
#     opt = RMSprop(1e-5)
#     m.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
#     m.summary()
#     return m

def model_define():
    ip = Input(shape=(10, 10, 1))
    f0 = 32
    h = Conv2D(f0, 5, activation='relu', padding='same')(ip)

    h = Conv2D(f0, 3, padding='same')(h) 
    # h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Conv2D(f0*2, 3, strides=(2, 2), padding='same')(h)
    # h = BatchNormalization(scale=False)(h)
    h = Activation('relu')(h)    

    h = Conv2D(f0*2, 3, padding='same')(h)
    # h = BatchNormalization(scale=False)(h)
    h = Activation('relu')(h)

    # h = Conv2D(f0*2, 3, strides=(2, 2), padding='same')(h)
    # # h = BatchNormalization(scale=False)(h)
    # h = Activation('relu')(h)    

    # h = Conv2D(f0*2, 1, padding='same')(h)
    # # h = BatchNormalization(scale=False)(h)
    # h = Activation('relu')(h) 

    h = Flatten()(h)

    h = Dense(8)(h)
    # h = BatchNormalization(scale=False)(h)
    h = Activation('relu')(h)
    h = Dropout(0.75)(h)
    out = Dense(2, activation='sigmoid')(h)
    m = Model(ip, out)
    opt = RMSprop(1e-3, decay=0.001)
    m.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    m.summary()
    return m


def model_define_():
    ip = Input(shape=(10, 10, 1))
    reps = []
    h = Conv2D(8, 3, activation='elu', padding='same')(ip)
    h = Conv2D(8, 1, activation='elu', padding='same')(h)
    h = Conv2D(16, 3, strides=(2, 2), activation='elu', padding='same')(h)
    h = Conv2D(16, 1, activation='elu', padding='same')(h)
    h = Flatten()(h)
    h = Dense(256, activation='elu')(h)
    reps.append(h)

    h = Conv2D(8, 3, activation='elu', padding='same')(ip)
    h = Conv2D(8, 1, activation='elu', padding='same')(h)
    h = Conv2D(16, 3, strides=(2, 2), activation='elu', padding='same')(h)
    h = Conv2D(16, 1, activation='elu', padding='same')(h)
    h = Conv2D(16, 3, activation='elu', padding='same')(h)
    h = Conv2D(32, 3, strides=(2, 2), activation='elu', padding='same')(h)
    h = Conv2D(32, 1, activation='elu', padding='same')(h)
    h = Flatten()(h)
    h = Dense(256, activation='elu')(h)
    reps.append(h)

    h = Conv2D(8, 3, activation='elu', padding='same')(ip)
    h = Conv2D(8, 1, activation='elu', padding='same')(h)
    h = Conv2D(16, 3, strides=(2, 2), activation='elu', padding='same')(h)
    h = Conv2D(16, 1, activation='elu', padding='same')(h)
    h = Conv2D(32, 3, strides=(2, 2), activation='elu', padding='same')(h)
    h = Conv2D(32, 1, activation='elu', padding='same')(h)
    h = Conv2D(64, 3, strides=(2, 2), activation='elu', padding='same')(h)
    h = Conv2D(64, 1, activation='elu', padding='same')(h)
    h = Flatten()(h)
    h = Dense(256)(h)
    reps.append(h)

    h = Flatten()(ip)
    h = Dense(128, activation='elu')(h)
    h = Dense(128, activation='elu')(h)
    h = Dense(256, activation='elu')(h)
    h = Dense(256, activation='elu')(h)
    reps.append(h)

    h = concatenate(reps)
    h0 = Dense(1024, activation='elu')(h)
    h = Dense(1024, activation='elu')(h0)
    h = Dense(1024, activation='elu')(h)
    h = add([h0, h])
    h = Dense(1024, activation='elu')(h)
    h = Dropout(0.5)(h)
    out = Dense(2, activation='sigmoid')(h)

    m = Model(ip, out)
    opt = RMSprop(1e-3)
    m.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    m.summary()
    return m


def predict(m, data, mode):
    print('Predict ' + mode)
    pred = m.predict(data[0])
    print(pred)
    print(data[1])
    # err = np.abs(poss - pred)
    # for t, p, e in zip(poss, pred, err):
    #     print("%0.5f\t%0.5f\t%0.5f" % (t, p, e))
    # pred_all = m.predict([data[0][0], data[0][1]])
    # pos_all = data[1]
    # err_all = pos_all - pred_all
    # np.save('err' + mode + '.npy', err_all)
    # np.save('img' + mode + '.npy', imgs)
    # np.save('pred' + mode + '.npy', pred)
    # np.save('poss' + mode + '.npy', poss)


import tensorflow as tf
import keras.backend as K
if __name__ == "__main__":
    m = model_define()
    tf.summary.FileWriter('./log', K.get_session().graph)
    train_data, test_data = load_data()
    if len(sys.argv) > 1:
        if sys.argv[1] == 'p':
            load_step = int(sys.argv[2])
            m.load_weights('save-%d' % load_step)
            predict(m, train_data, 'train')
            predict(m, test_data, 'test')
        if sys.argv[1] == 't':
            load_step = int(sys.argv[2])
            m.load_weights('save-%d' % load_step)
            train(m, train_data, test_data)
    else:
        train(m, train_data, test_data)
