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

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_train = "train.csv"  # 训练文件
path_test = "test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train, header=None)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
            "CALLSTATE", "Y"]


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

    row[0] = row[0] % 86400 / (60 * 5)
    row[1] = maxMinNormalization(row[1], 80, 130)
    row[2] = maxMinNormalization(row[2], 20, 45)
    row[3] = maxMinNormalization(row[3], 0, 360)
    row[4] = maxMinNormalization(row[4], -300, 3500)
    row[5] = maxMinNormalization(row[5], 0, 60)
    row[6] = maxMinNormalization(row[6], 0, 4)

    return row

def fc2():
    model = Sequential()
    model.add(Dense(32, input_shape=(7,)))
    model.add(Activation('relu'))
    model.add(Dense(1))

def train(model):
    model = fc2()
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

def predict(model):
    score = model.predict(X_test, batch_size=32, verbose=1)

def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    import numpy as np

    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                writer.writerow([item[0], np.random.rand()]) # 随机值

                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    process()

