from keras.layers import Conv2D, Dense, Flatten, ELU, Input, concatenate, Dropout, MaxPool2D, add
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
import numpy as np
import sys
import random


def load_data():
    pho = np.load(
        '/home/hongxwing/Workspace/Datas/DetectorCalibration/data_all/pho_c.npy')
    pos = np.load(
        '/home/hongxwing/Workspace/Datas/DetectorCalibration/data_all/pos_c.npy')
    cen = np.load(
        '/home/hongxwing/Workspace/Datas/DetectorCalibration/data_all/cen_c.npy')
    ipt = np.reshape(pho, [-1, 10, 10, 1])
    m = np.mgrid[-27:29:6, -27:28:6]
    m.shape
    x = m[1]
    y = m[0]
    ipt2 = np.zeros([ipt.shape[0], 10, 10, 2])
    ipt2[:, :, :, 0]  = ipt[:, :, :, 0]
    for i in range(ipt2.shape[0]):
        ipt2[i, :, :, 1] = x
        # ipts[i, :, :, 2] = y
    ipt = np.array(ipt2)
    # ipt = np.log(ipt)
    # ipt = ipt - np.mean(ipt)
    # ipt = ipt / np.std(ipt)
    x = pos[:, 0]
    x = np.reshape(x, [-1, 1])
    cx = cen[:, 0]
    cx = np.reshape(cx, [-1, 1])
    nb_train = ipt.shape[0] // 5 * 4
    idx = list(range(x.shape[0]))
    # random.shuffle(idx)
    ipt_train = ipt[idx[:nb_train], :]
    x_train = x[idx[:nb_train], :]
    ipt_test = ipt[idx[nb_train:], :]
    x_test = x[idx[nb_train:], :]
    cx_train = cx[idx[:nb_train], :]
    cx_test = cx[idx[nb_train:], :]
    # return ((ipt_train, cx_train), x_train), ((ipt_test, cx_test), x_test)
    return (ipt_train, x_train), (ipt_test, x_test)

# from xlearn.utils.general import enter_debug


def train(m, train_data, test_data):
    # enter_debug()
    nb_epochs = 100
    for i in range(10):
        print("TRAING %d" % i)
        # m.fit(list(train_data[0]), train_data[1], batch_size=128,
        #       epochs=nb_epochs, validation_data=test_data)
        print(train_data[0].shape)
        print(train_data[1].shape)
        m.fit(train_data[0], train_data[1], batch_size=128,
              epochs=nb_epochs, validation_data=test_data)
        m.save('save-%d' % i)
        pdx = m.predict(test_data[0])
        lbx = test_data[1]
        err = lbx - pdx
        # err_s = np.sqrt(np.mean(np.square(err)))
        err_s = np.mean(np.abs(err))
        print('err_test', err_s)


def model_define():
    ip = Input(shape=(10, 10, 2))
    h = Flatten()(ip)
    h = Dense(10, activation='relu')(h)
    h = Dense(8, activation='relu')(h)
    out = Dense(1)(h)
    m = Model(ip, out)
    opt = RMSprop(1e-7)
    m.compile(loss='mse', optimizer=opt)
    m.summary()
    return m

def model_define_1():
    is_cata = False
    reps = []
    ip = Input(shape=(10, 10, 2))
    ipc = Input(shape=(1,))
    h = Conv2D(32, 3, activation='elu')(ip)
    h = MaxPool2D()(h)
    reps.append(Flatten(name='rep0')(h))

    h = Conv2D(128, 3, activation='elu')(h)
    h = MaxPool2D()(h)
    h = Dropout(0.5)(h)
    # h = Conv2D(256, 3, activation='elu')(h)
    # h = Dropout(0.5)(h)
    # h = Conv2D(512, 3, activation='elu')(h)
    reps.append(Flatten(name='rep1')(h))

    h = Conv2D(8, 3, activation='elu', padding='same')(ip)
    h = MaxPool2D()(h)
    h = Conv2D(32, 3, activation='elu', padding='same')(h)
    h = MaxPool2D()(h)
    h = Conv2D(128, 3, activation='elu', padding='same')(h)
    h = MaxPool2D()(h)
    h = Conv2D(512, 1, activation='elu', padding='same')(h)
    h = Dropout(0.5)(h)
    h = Flatten(name='rep2')(h)
    reps.append(h)

    h = Conv2D(8, 3, activation='elu')(ip)
    h = Conv2D(16, 3, activation='elu')(h)
    h = Conv2D(32, 3, activation='elu')(h)
    h = Conv2D(64, 3, activation='elu')(h)
    h = Conv2D(64, 1, activation='elu')(h)
    h = Dropout(0.5)(h)
    reps.append(Flatten(name='rep3')(h))

    h = Conv2D(8, 5, activation='elu', padding='same')(ip)
    h = MaxPool2D()(h)
    h = Conv2D(32, 5, activation='elu', padding='same')(h)
    h = MaxPool2D()(h)
    h = Conv2D(64, 1, activation='elu')(h)
    h = Dropout(0.5)(h)
    reps.append(Flatten(name='rep4')(h))

    h = Conv2D(32, 5, activation='elu')(ip)
    h = Conv2D(64, 5, activation='elu')(h)
    h = Conv2D(64, 1, activation='elu')(h)
    h = Dropout(0.5)(h)
    reps.append(Flatten(name='rep5')(h))

    h = Flatten()(ip)
    reps.append(h)
    for i in range(2):
        h = Dense(128, activation='elu')(h)
    h = Dropout(0.5)(h)
    reps.append(Dense(128, activation='elu', name='rep6')(h))

    h = Conv2D(8, 5, activation='elu', padding='same')(ip)
    h = MaxPool2D(pool_size=(1, 2))(h)
    h = Conv2D(16, 5, activation='elu', padding='same')(h)
    h = MaxPool2D(pool_size=(1, 2))(h)
    h = Conv2D(32, 1, activation='elu')(h)
    h = Dropout(0.5)(h)
    reps.append(Flatten(name='rep7')(h))

    reps.append(ipc)
    h = concatenate(reps)
    h = Dense(1024, activation='elu')(h)
    h = Dropout(0.5)(h)
    h = Dense(1024, activation='elu')(h)
    h = Dropout(0.5)(h)
    out = Dense(1)(h)
    out = add([out, ipc])
    m = Model([ip, ipc], out)
    opt = Adam(lr=1e-3)
    m.compile(loss='mse', optimizer=opt)
    m.summary()
    return m


def predict(m, data, mode):
    print('Predict ' + mode)
    data = list(data)
    # data[0][0] = data[0][0][:10000, ...]
    # data[0][1] = data[0][1][:10000, ...]
    # data[1] = data[1][:10000, :]
    idx = list(range(data[0][0].shape[0]))
    # random.shuffle(idx)
    idx = idx[:32]
    imgs = data[0][0][idx, :, :, :]
    cxs = data[0][1][idx, :]
    poss = data[1][idx, :]
    pred = m.predict([imgs, cxs])
    err = np.abs(poss - pred)
    for t, p, e in zip(poss, pred, err):
        print("%0.5f\t%0.5f\t%0.5f" % (t, p, e))
    pred_all = m.predict([data[0][0], data[0][1]])
    pos_all = data[1]
    err_all = pos_all - pred_all
    np.save('err' + mode + '.npy', err_all)
    np.save('img' + mode + '.npy', imgs)
    np.save('pred' + mode + '.npy', pred)
    np.save('poss' + mode + '.npy', poss)


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
