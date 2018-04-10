# -*- coding:utf8 -*-
from __future__ import print_function
import os,sys
import csv
from collections import Counter
import time
import pandas as pd
import numpy as np

path_train = "/data/dm/train.csv"
path_test = "/data/dm/test.csv"

debug = False
if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    debug = True
    path_train = "train.csv"
    path_test = "test.csv"

path_test_out = "model/"
stat_y = []
class_num = 20
class_thr = 0.5
zero_sample_rate = 0.1
def normalize(row):
    """
    time      timestamp % 86400 / (60 * 5)
    longitude 82.82367616 - 127.4035967
    latitude  21.57433768 - 44.000393
    direction 0 - 360
    height    -271.488617 - 3411.894775
    speed     0 - 53.48
    callstate 0 - 4
    """
    def minmaxnormalization(x, min, max):
        x = (x - min) / (max - min)
        return x

    row[0] = minmaxnormalization(int(row[0]) % 86400 / (60 * 5), 0, 288)
    row[1] = minmaxnormalization(float(row[1]), 80, 130)
    row[2] = minmaxnormalization(float(row[2]), 20, 45)
    row[3] = minmaxnormalization(float(row[3]), 0, 360)
    row[4] = minmaxnormalization(float(row[4]), -300, 3500)
    row[5] = minmaxnormalization(float(row[5]), 0, 60)
    row[6] = minmaxnormalization(float(row[6]), 0, 4)

    return row

def data_reader(path, is_pred=False):
    def padding(x, size=700):
        if len(x) < size:
            x += (size - len(x)) * [[0,0,0,0,0,0,0]]
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
                        yield (x, yy)
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

def gini_test(y_pred) :
    n_samples = y_pred.shape[0]

    pred_order = y_pred.astype(float)

    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    print(L_ones[0])
    G_pred = np.sum(L_ones - L_pred)
    print(G_pred)

def predict():
    print('Predict...')
    scores = []
    tot_user = 0
    for x_pred, Id in data_reader(path_test, is_pred=True):
        tot_user += 1
    for x_pred, Id in data_reader(path_test, is_pred=True):
        score = 1.0/tot_user
        if debug:
            print(Id, score)
        scores.append([Id, score])
    gini_test(np.array(scores)[:,1])
    return scores

def output(scores):
    print('Output...')
    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for Id, score in scores:
            writer.writerow([Id, score])

def process():
    scores = predict()
    output(scores)
    print('Done')

if __name__ == "__main__":
    print('Main...')
    start_time = time.time()
    process()
    print(Counter(stat_y))
    print ('tot_time:',time.time() - start_time)
