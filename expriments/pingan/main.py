# -*- coding:utf8 -*-
from __future__ import print_function
import os
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
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

def fit_reader():
    data = np.array(read_csv(path_train))
    label = data[1:, -1]
    data = data[1:, [1,3,4,5,6,7,8]]
    data = np.array([normalize(row) for row in data])
    return data, label

def fc2():
    model = Sequential()
    model.add(Dense(32, input_shape=(7,)))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()
    return model

def train():
    print('Train...')
    x_train, y_train = fit_reader()
    print(x_train.shape[0], 'train samples')
    model = fc2()
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=256, epochs=200, verbose=1)
    return model

def predict(model):
    print('Predict...')
    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue

                Id = item[0]
                item = np.array([normalize(np.array(item)[[1,3,4,5,6,7,8]])])
                score = model.predict(item)
                writer.writerow([Id, score[0]])

                ret_set.add(Id)

def process():
    model = train()
    predict(model)
    print('Done')

if __name__ == "__main__":
    print("****************** start **********************")
    process()
