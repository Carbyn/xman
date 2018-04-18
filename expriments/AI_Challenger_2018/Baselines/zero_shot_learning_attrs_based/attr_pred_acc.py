#!/usr/bin/env python
# coding=utf-8

"""
python pred.py Animals model/mobile_Animals_wgt.h5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Activation
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

eval_acc_result = {}
def eval_acc(y_pred, ground_truth, abs_error=0.1):
    if len(y_pred) != len(ground_truth):
        print('eval_acc: len(y_pred)=%d len(ground_truth)=%d' % (len(y_pred), len(ground_truth)))
        exit()
    for i,val in enumerate(ground_truth):
        if str(i) not in eval_acc_result.keys():
            eval_acc_result[str(i)] = 0
        if abs(val - y_pred[i]) <= abs_error:
            eval_acc_result[str(i)] += 1

def main(superclass, model_weight, img_path, model=None):
    is_debug = False
    classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    classAttrsNums = {'Animals': 123, 'Fruits': 58}
    classAttrsNum = classAttrsNums[superclass]
    date = '20180321'

    class_attrs_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_attributes_per_class_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
    train_labels_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_labels_%s.txt' % (superclass.lower(), date, superclass.lower(), date)

    fattrs = open(class_attrs_path, 'r', encoding='utf-8')
    attrs = fattrs.readlines()
    fattrs.close()
    label_attrs_train = {} #{'label': [attr]}
    for row in attrs:
        pair = row.strip().split(',')
        label_attrs_train[pair[0]] = list(map(lambda x: float(x), pair[1].strip().split(' ')[1:-1]))

    ftrain = open(train_labels_path, 'r', encoding='utf-8')
    labels = ftrain.readlines()
    ftrain.close()
    train_labels = {} #{'img': [attr]}
    for row in labels:
        parts = row.strip().split(', ')
        train_labels[parts[-1].split('/')[1]] = label_attrs_train[parts[1]]

    Y = {}
    img = image.load_img(img_path, target_size=(72, 72))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    x = x/255.0
    #print(np.shape(x))
    #print(x[0][0][0])
    y_pred = model.predict(x)
    y_pred = y_pred[0]
    y_pred = list(map(lambda x: round(x,2), y_pred))
    eval_acc(y_pred, train_labels[img_path.split('/')[-1]])

def stat(val_acc_path):
    with open(val_acc_path, 'r') as reader:
        c = {}
        thrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for row in reader.readlines():
            pair = row.strip().split(' ')
            for thr in thrs:
                if thr not in c.keys():
                    c[thr] = 0
                if float(pair[1]) <= thr:
                    print(pair[0])
                    c[thr] += 1

        for thr in thrs:
            print('thr=%s percentage=%f' % (thr, round(c[thr]/1.23,1)))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        superclass = sys.argv[1]
        model_weight = sys.argv[2]

        val_acc_path = 'attr_pred_acc_%s.txt' % (superclass.lower())
        #stat(val_acc_path)
        #exit()

        date = '20180321'
        val_dir = 'trainval_%s/val' % (superclass)
        classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
        classAttrsNums = {'Animals': 123, 'Fruits': 58}
        classAttrsNum = classAttrsNums[superclass]

        base_model = Xception(include_top=True, weights=None,
                          input_tensor=None, input_shape=(72,72,3),
                          pooling=None, classes=classNum[superclass[0]])
        output = Dense(classAttrsNum, activation='sigmoid', name='predictions')(base_model.get_layer('avg_pool').output)
        model = Model(inputs=base_model.input, outputs=output)
        model.load_weights(model_weight)

        count = 0
        classes = os.listdir(val_dir)
        for c in classes:
            if c[0] == '.':
                continue
            classpath = val_dir + '/' + c
            images = os.listdir(classpath)
            for img in images:
                if img[0] == '.':
                    continue
                img_path = classpath + '/' + img
                label = main(superclass, model_weight, img_path, model)
                count += 1
       
        fsave = open(val_acc_path, 'w')
        eval_acc_result_final = []
        for k,v in eval_acc_result.items():
            eval_acc_result_final.append(v/count)
            print(k, v/count)
            fsave.write(str(k)+' '+str(v/count)+'\n')
        fsave.close()

        stat(val_acc_path)

    else:
        print('Param error')
        exit()
