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
import numpy as np
import sys
import os

def main():
    if len(sys.argv) == 4:
        superclass = sys.argv[1]
        model_weight = sys.argv[2]
        img_path = sys.argv[3]
    else:
        print('Param error')
        exit()

    classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    classAttrsNums = {'Animals': 123, 'Fruits': 58}
    classAttrsNum = classAttrsNums[superclass]
    unknown_labels = {
        'Animals': [
            'Label_A_02',
            'Label_A_05',
            'Label_A_08',
            'Label_A_14',
            'Label_A_20',
            'Label_A_29',
            'Label_A_31',
            'Label_A_35',
            'Label_A_39',
            'Label_A_41'
        ],
        'Fruits': [
            'Label_F_03',
            'Label_F_09',
            'Label_F_10',
            'Label_F_17',
            'Label_F_25',
            'Label_F_29',
            'Label_F_31',
            'Label_F_34',
            'Label_F_43',
            'Label_F_49'
        ]
    }
    date = '20180321'

    class_attrs_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_attributes_per_class_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
    attrs_list_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_attribute_list_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
    test_dir= '../zsl_a_%s_test_%s/' % (superclass.lower(), date)
    pred_path = 'pred_%s.txt' % (superclass)

    fattrs = open(class_attrs_path, 'r', encoding='utf-8')
    attrs = fattrs.readlines()
    fattrs.close()
    label_attrs = {}
    for row in attrs:
        pair = row.strip().split(',')
        if pair[0] not in unknown_labels[superclass]:
            continue
        label_attrs[pair[0]] = list(map(lambda x: float(x), pair[1].strip().split(' ')[1:-1]))


    base_model = Xception(include_top=True, weights=None,
                      input_tensor=None, input_shape=(72,72,3),
                      pooling=None, classes=classNum[superclass[0]])
    output = Dense(classAttrsNum, activation='sigmoid', name='predictions')(base_model.get_layer('avg_pool').output)
    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights(model_weight)

    Y = {}
    img = image.load_img(img_path, target_size=(72, 72))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    x = x/255.0
    print(np.shape(x))
    print(x[0][0][0])
    y_pred = model.predict(x)
    y_pred = y_pred[0]
    y_pred = list(map(lambda x: round(x,2), y_pred))
    y_pred = np.array(y_pred)
    with open(attrs_list_path, 'r') as f:
        attrs_map = [] 
        for line in f.readlines():
            tokens = line.split(', ')
            attrs_map.append(tokens[-1])
        attrs_map = np.array(attrs_map)
    print(attrs_map[np.where(y_pred>0.5)])


    date = '20180321'
    superclass = 'Animals'
    class_attrs_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_attributes_per_class_%s.txt' % (superclass.lower(), date, superclass.lower(), date)

    fattrs = open(class_attrs_path, 'r', encoding='utf-8')
    attrs = fattrs.readlines()
    fattrs.close()
    label_attrs = {}
    for row in attrs:
        pair = row.strip().split(',')
        if pair[0] not in unknown_labels[superclass]:
            continue
        label_attrs[pair[0]] = list(map(lambda x: float(x), pair[1].strip().split(' ')[1:-1]))
    nearest = 0.0
    y_label = ''
    for label, attr in label_attrs.items():
        #dist = distance.euclidean(attr, y_pred)
        #dist = distance.hamming(attr, y_pred)
        #dist = 1-distance.pdist(np.vstack([attr, y_pred]),'cosine')

        attr = np.array(attr)
        y_pred = np.array(y_pred)
        y_pred_thr = 0.5
        attr_count = np.shape(np.where(attr>0))[1]
        y_pred_count = np.shape(np.where(y_pred>y_pred_thr))[1]
        union = np.shape(np.where(attr[np.where(y_pred>y_pred_thr)] > 0))[1]
        dist = -union/(attr_count+y_pred_count-union)
        print(label, dist)
        if nearest == 0:
            nearest = dist
            y_label = label
        if dist < nearest:
            nearest = dist
            y_label = label
        print(y_label)
if __name__ == "__main__":
    main()
