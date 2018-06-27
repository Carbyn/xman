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
from utils import *
import numpy as np
import sys
import os
def main(superclass, model_weight, img_path, model=None):
    is_debug = False
    classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    classAttrsNums = {'Animals': 123, 'Fruits': 58}
    classAttrsNums = {'Animals': 99, 'Fruits': 48}
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
    attr_acc_path = 'attr_pred_acc_%s.txt' % (superclass.lower())

    facc = open(attr_acc_path, 'r', encoding='utf-8')
    acc = facc.readlines()
    facc.close()
    acc = [float(row.strip().split(' ')[1]) for row in acc]


    if model == None:
        is_debug = True
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
    #print(np.shape(x))
    #print(x[0][0][0])
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
    if is_debug:
        print(attrs_map[np.where(y_pred>0.5)])


    date = '20180321'
    class_attrs_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_attributes_per_class_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
    to_be_removed_attrs = [3, 16, 18, 20, 21, 27, 36, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 66, 96, 106, 107, 108, 119]
    to_be_removed_attrs = [2, 5, 18, 21, 23, 25, 29, 33, 45, 46]

    fattrs = open(class_attrs_path, 'r', encoding='utf-8')
    attrs = fattrs.readlines()
    fattrs.close()
    label_attrs = {}
    label_attrs_list = [] 
    for row in attrs:
        pair = row.strip().split(',')
        if pair[0] not in unknown_labels[superclass]:
            continue
        label_attrs[pair[0]] = list(map(lambda x: float(x), pair[1].strip().split(' ')[1:-1]))
        label_attrs[pair[0]] = list(np.delete(np.array(label_attrs[pair[0]]), to_be_removed_attrs))
        label_attrs_list.append(label_attrs[pair[0]])
    attrs_entropy = calc_attrs_entropy(label_attrs)

    label_attrs_weight = np.sum(np.array(label_attrs_list),axis=0)
    label_attrs_weight = 1 - label_attrs_weight/10.0
    #label_attrs_weight[26:29] = 10 # just for fun
    #label_attrs_weight[:] = 1 
    nearest = 0.0
    y_label = ''
    y_pred = np.array(y_pred)*np.array(acc)
    for label, attr in label_attrs.items():
        attr = np.array(attr)
        topK = 90 
        y_pred_index = np.argsort(y_pred)
        attrs_sum = np.sum(label_attrs_weight[np.where(attr > 0)])
        pred_sum = np.sum(label_attrs_weight[y_pred_index])
        union_sum = sum([i*y_pred[index] for index, i in enumerate(label_attrs_weight) if index in y_pred_index and attr[index]> 0])
        dist = -(union_sum)/(pred_sum + attrs_sum - union_sum)
        dist = -(union_sum)
        if is_debug:
            print(label, dist)
        if nearest == 0:
            nearest = dist
            y_label = label
        if dist < nearest:
            nearest = dist
            y_label = label
    if is_debug:
        print(y_label)
    return y_label
if __name__ == "__main__":
    if len(sys.argv) == 4:
        superclass = sys.argv[1]
        model_weight = sys.argv[2]
        img_path = sys.argv[3]
        main(superclass, model_weight, img_path)
    elif len(sys.argv) == 3:
        superclass = sys.argv[1]
        model_weight = sys.argv[2]
        pred_path = 'pred_%s.txt' % (superclass)
        date = '20180321'
        test_dir= '../zsl_a_%s_test_%s/' % (superclass.lower(), date)
        classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
        classAttrsNums = {'Animals': 123, 'Fruits': 58}
        classAttrsNums = {'Animals': 99, 'Fruits': 48}
        classAttrsNum = classAttrsNums[superclass]
        base_model = Xception(include_top=True, weights=None,
                          input_tensor=None, input_shape=(72,72,3),
                          pooling=None, classes=classNum[superclass[0]])
        output = Dense(classAttrsNum, activation='sigmoid', name='predictions')(base_model.get_layer('avg_pool').output)
        model = Model(inputs=base_model.input, outputs=output)
        model.load_weights(model_weight)
        Y = {}
        images = os.listdir(test_dir)
        for c,i in enumerate(images):
            if i[0] == '.':
                continue
            print(c,i)
            img_path = test_dir + '/' + i
            label = main(superclass, model_weight, img_path, model)
            Y[i] = label
        fpred = open(pred_path, 'w', encoding='utf-8')
        fpred.write('\n'.join([image+' '+label for image,label in Y.items()]))
        fpred.close()


    else:
        print('Param error')
        exit()
