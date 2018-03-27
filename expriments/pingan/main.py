# -*- coding:utf8 -*-
from __future__ import print_function
import os
import csv
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD
from keras import losses

path_train = "/data/dm/train.csv"
path_test = "/data/dm/test.csv"

path_train = "train.csv"
path_test = "test.csv"

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

def fit_rnn_reader():
    data = np.array(read_csv(path_train))[1:]
    dct = {}
    for item in data:
        if item[0] not in dct.keys():
            dct[item[0]] = []
        dct[item[0]].append(np.append(normalize(np.array(item)[[1,3,4,5,6,7,8]]), float(item[-1])))
    data = np.array(list(dct.values()))
    return data[:,:,:-1], data[:,:,-1]

def predict_rnn_reader():
    from collections import defaultdict
    data = np.array(read_csv(path_test))[1:]
    dct = defaultdict(list)
    for item in data:
        dct[item[0]].append(normalize(np.array(item)[[0,1,3,4,5,6,7,8]]))
    data = np.array(dct.values())
    return data

def gru2(rnn_size=512):
    input_data = Input(name='the_input', shape=(None, 7), dtype='float32')
    inner = Dense(32, activation='relu', name='inner')(input_data)
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    output = Dense(1, kernel_initializer='he_normal', name='dense2')(concatenate([gru_2, gru_2b]))
    model = Model(inputs=input_data, outputs=output) 
    model.summary()
    return model 

def train():
    print('Train...')
    x_train, y_train = fit_rnn_reader()
    print(x_train.shape[0], 'train samples')
    model = gru2()
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=256, epochs=20, verbose=1)
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
    x_pred = predict_rnn_reader()
    scores = []
    for item in x_pred:
        score = model.predict(item[:, 1:])
        scores.append([item[0][0], score])
    output(scores)

def process():
    model = train()
    predict(model)
    print('Done')

if __name__ == "__main__":
    print("****************** start **********************")
    process()
