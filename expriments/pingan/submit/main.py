# -*- coding:utf8 -*-
from __future__ import print_function
import os,sys
import csv
from collections import Counter
import time
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Masking, Flatten
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization 
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras import losses

path_train = "/data/dm/train.csv"
path_test = "/data/dm/test.csv"

debug = False
if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    debug = True
    path_train = "train.csv"
    path_test = "test.csv"

path_test_out = "model/"
stat_y = [] 
stat_y_pred = []
class_num = 1
if debug:
    class_num = 1
class_thr = 0.5
zero_sample_rate = 0.05
zero_sample_rate = 1
def normalize(row):
    """
    TIME      timestamp % 86400 / (60 * 5)
    LONGITUDE 82.82367616 - 127.4035967
    LATITUDE  21.57433768 - 44.000393
    DIRECTION 0 - 360
    HEIGHT    -271.488617 - 3411.894775
    SPEED     0 - 53.48
    CALLSTATE 0 - 4
    """
    def minMaxNormalization(x, Min, Max):
        x = (x - Min) / (Max - Min)
        return x

    row[0] = minMaxNormalization(int(row[0]) % 86400 / (60 * 5), 0, 288)
    row[1] = minMaxNormalization(float(row[1]), 80, 130)
    row[2] = minMaxNormalization(float(row[2]), 20, 45)
    row[3] = minMaxNormalization(float(row[3]), 0, 360)
    row[4] = minMaxNormalization(float(row[4]), -300, 3500)
    row[5] = minMaxNormalization(float(row[5]), 0, 60)
    row[6] = minMaxNormalization(float(row[6]), 0, 4)

    return row

def data_reader(path, is_pred=False):
    def padding(x, size=700):
        if len(x) < size:
            x += (size - len(x)) * [normalize([144,105,32.5,180,1000,30,0])]
        return x[:size]

    epoch_cnt = 0 

    while 1:
        f = open(path)
        i = 0
        cur_id = 0
        x = []
        y = 0
        for line in f:
            i += 1
            if i == 1:
                continue
            item = line.split(',')
            if cur_id == 0:
                cur_id = item[0]
            if item[0] != cur_id:
                x = padding(x)
                if is_pred:
                    yield (x, cur_id)
                else:
                    yy = np.zeros(class_num)
                    if epoch_cnt == 0:
                        stat_y.append(int(round(y/class_thr))) 
                    max_y = min(class_num - 1, int(round(y/class_thr)))
                    if not (max_y == 0 and np.random.random() > zero_sample_rate):
                        yy[max_y] = 1
                        yield (x, y)
                x = []
                cur_id = item[0]
            x.append(normalize(np.array(item)[[1,3,4,5,6,7,8]]))
            if not is_pred:
                y = float(item[-1])
        epoch_cnt+=1
        f.close()
        if is_pred:
            yield (padding(x), cur_id)
            break

def fit_data_reader(batch_size=256):
    X = []
    Y = []
    for x,y in data_reader(path_train):
        X.append(x)
        Y.append(y)
        if len(X) == batch_size:
            yield ({'input': np.array(X)}, {'output': np.array(Y)})
            X = []
            Y = []

def gru(rnn_size=16, depth=7):
    input_data = Input(name='input', shape=(700, 7), dtype='float32')
    inner = Dense(8, activation='relu', name='inner')(input_data)
    for i in range(depth):
        gru_forward = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal')(inner)
        gru_backward = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(inner)
        inner = add([gru_forward, gru_backward])

    flatten = Flatten()(concatenate([gru_forward, gru_backward]))
    #bn = BatchNormalization()(flatten)
    #dense_1 = Dense(128, kernel_initializer='he_normal', name='dense_1', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(bn)
    #dense_2 = Dense(32, kernel_initializer='he_normal', name='dense_2', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(dense_1)
    dense_1 = Dense(128, kernel_initializer='he_normal', name='dense_1')(flatten)
    dense_2 = Dense(16, kernel_initializer='he_normal', name='dense_2')(dense_1)
    output = Dense(class_num, kernel_initializer='he_normal', name='output')(dense_2)
    model = Model(inputs=input_data, outputs=output)
    if debug:
        model.summary()
    return model

def train():
    print('Train...')
    model = gru()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    batch_size = 128
    steps_per_epoch = 32
    epochs = 10
    if debug:
        batch_size = 32
        steps_per_epoch = 32
        epochs = 2
    model.fit_generator(fit_data_reader(batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)
    return model

def predict(model):
    print('Predict...')
    scores = []
    for x_pred, Id in data_reader(path_test, is_pred=True):
        score = model.predict(np.array([x_pred]))
        score = score[0][0]
        #score = np.argmax(score[0])*class_thr
        score = max(0, score)
        stat_y_pred.append(int(score/0.25))
        if debug:
            print(Id, score)
        scores.append([Id, score])
    return scores

def output(scores):
    print('Output...')
    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for Id, score in scores:
            writer.writerow([Id, score])

def process():
    model = train()
    scores = predict(model)
    output(scores)
    print('Done')

if __name__ == "__main__":
    print('Main...')
    start_time = time.time()
    process()
    #print(Counter(stat_y))
    print(Counter(stat_y_pred))
    print ('tot_time:',time.time() - start_time)
