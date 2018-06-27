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
from utils import *
import numpy as np
import sys
import os

def main(superclass, model_weight):
    date = '20180321'
    classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    classAttrsNums = {'Animals': 123, 'Fruits': 58}
    unseen_labels = {
        'Animals': ['Label_A_02', 'Label_A_05', 'Label_A_08', 'Label_A_14', 'Label_A_20', 'Label_A_29', 'Label_A_31', 'Label_A_35', 'Label_A_39', 'Label_A_41'],
        'Fruits': ['Label_F_03', 'Label_F_09', 'Label_F_10', 'Label_F_17', 'Label_F_25', 'Label_F_29', 'Label_F_31', 'Label_F_34', 'Label_F_43', 'Label_F_49']
    }

    class_attrs_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_attributes_per_class_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
    attr_acc_path = 'attr_pred_acc_%s.txt' % (superclass.lower())
    low_entropy_attrs_path = 'low_entropy_attrs_%s.txt' % (superclass.lower())
    test_dir= '../zsl_a_%s_test_%s/' % (superclass.lower(), date)
    pred_path = 'pred_%s.txt' % (superclass)

    facc = open(attr_acc_path, 'r', encoding='utf-8')
    attr_acc = facc.readlines()
    facc.close()
    attr_acc = [float(row.strip().split(' ')[1]) for row in attr_acc]

    #fent = open(low_entropy_attrs_path, 'r', encoding='utf-8')
    #low_entropy_attrs = fent.readline()
    #fent.close()
    #low_entropy_attrs = [int(x) for x in low_entropy_attrs.strip().split(' ')]

    #classAttrsNum = classAttrsNums[superclass] - len(low_entropy_attrs) 

    fattrs = open(class_attrs_path, 'r', encoding='utf-8')
    attrs = fattrs.readlines()
    fattrs.close()
    label_attrs = {}
    label_attrs_list = [] 
    for row in attrs:
        pair = row.strip().split(',')
        if pair[0] not in unseen_labels[superclass]:
            continue
        label_attrs[pair[0]] = list(map(lambda x: float(x), pair[1].strip().split(' ')[1:-1]))
        #label_attrs[pair[0]] = list(np.delete(np.array(label_attrs[pair[0]]), low_entropy_attrs))
        label_attrs_list.append(label_attrs[pair[0]])
    label_attrs_weight = np.sum(np.array(label_attrs_list),axis=0)
    label_attrs_weight = 1 - label_attrs_weight/10.0

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
        img_path = test_dir + '/' + i
        img = image.load_img(img_path, target_size=(72, 72))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        x = x/255.0
        y_pred = model.predict(x)
        y_pred = np.array(list(map(lambda x: round(x,2), y_pred[0])))

        nearest = 0.0
        y_label = ''
        y_pred = y_pred*np.array(attr_acc)
        for label, attr in label_attrs.items():
            attr = np.array(attr)
            topK = 90 
            y_pred_index = np.argsort(y_pred)
            attrs_sum = np.sum(label_attrs_weight[np.where(attr >= 1)])
            pred_sum = np.sum(label_attrs_weight[y_pred_index])
            union_sum = sum([i*y_pred[index] for index, i in enumerate(label_attrs_weight) if index in y_pred_index and attr[index]>=1])
            dist = -(union_sum)/(pred_sum + attrs_sum - union_sum)
            #dist = -(union_sum)
            if nearest == 0:
                nearest = dist
                y_label = label
            if dist < nearest:
                nearest = dist
                y_label = label
        Y[i] = label

    fpred = open(pred_path, 'w', encoding='utf-8')
    fpred.write('\n'.join([image+' '+label for image,label in Y.items()]))
    fpred.close()
    
if __name__ == "__main__":
    if len(sys.argv) == 3:
        superclass = sys.argv[1]
        model_weight = sys.argv[2]

        main(superclass, model_weight)

    else:
        print('Param error')
        exit()
