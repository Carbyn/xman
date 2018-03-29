# -*- coding:utf8 -*-
from __future__ import print_function
import os
import csv
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Masking, Flatten
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD
from keras import losses

path_train = "/data/dm/train.csv"
path_test = "/data/dm/test.csv"

#path_train = "train.csv"
#path_test = "test.csv"

path_test_out = "model/"

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

def read_csv(path):
    tempdata = pd.read_csv(path, header=None, low_memory=False)
    return tempdata

def data_reader(path, is_pred=False):
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
                if len(x) < 700:
                    x += (700 - len(x)) * [[0,0,0,0,0,0,0]]
                if is_pred:
                    yield (np.array(x)[:700], cur_id)
                else:
                    yield (x[:700], y)
                x = []
                cur_id = item[0]
            x.append(normalize(np.array(item)[[1,3,4,5,6,7,8]]))
            if not is_pred:
                y = float(item[-1])
        f.close()
        if is_pred:
            yield (np.array(x)[:700], cur_id)
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

def gru2(rnn_size=128):
    input_data = Input(name='input', shape=(700, 7), dtype='float32')
    #mask = Masking(mask_value=0., input_shape=(700, 7))(input_data)
    inner = Dense(32, activation='relu', name='inner')(input_data)
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    flatten = Flatten()(concatenate([gru_2, gru_2b]))
    output = Dense(1, kernel_initializer='he_normal', name='output')(flatten)
    model = Model(inputs=input_data, outputs=output)
    model.summary()
    return model

def train():
    print('Train...')
    model = gru2()
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.fit_generator(fit_data_reader(32), steps_per_epoch=50000, epochs=1, verbose=1)
    return model

def output(scores):
    print('Output...')
    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for Id, score in scores:
            writer.writerow([Id, score])

def predict(model):
    print('Predict...')
    scores = []
    for x_pred, Id in data_reader(path_test, is_pred=True):
        print('In...')
        score = model.predict(np.array([x_pred]))
        print(Id, score[0][0])
        scores.append([Id, score[0][0]])
    output(scores)

def process():
    model = train()
    predict(model)
    print('Done')

if __name__ == "__main__":
    print("****************** start **********************")
    process()
