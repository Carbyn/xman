#!/usr/bin/env python
# coding=utf-8

"""
python pred.py Animals
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import os

def calc_entropy(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def calc_attrs_entropy(label_attrs):
    attrs = np.array(list(label_attrs.values()))
    attrs_entropy = []
    for i in range(attrs.shape[1]):
        entropy = calc_entropy(attrs[:, i])
        if entropy > 0:
            entropy = 1.0
        else:
            entropy = 0
        attrs_entropy.append(entropy)
    return attrs_entropy

def calc_low_entropy_attrs(superclass):
    unseen_labels = {
        'Animals': ['Label_A_02', 'Label_A_05', 'Label_A_08', 'Label_A_14', 'Label_A_20', 'Label_A_29', 'Label_A_31', 'Label_A_35', 'Label_A_39', 'Label_A_41'],
        'Fruits': ['Label_F_03', 'Label_F_09', 'Label_F_10', 'Label_F_17', 'Label_F_25', 'Label_F_29', 'Label_F_31', 'Label_F_34', 'Label_F_43', 'Label_F_49']
    }
    date = 20180321
    class_attrs_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_attributes_per_class_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
    low_entropy_attrs_path = 'low_entropy_attrs_%s.txt' % (superclass.lower())
    fattrs = open(class_attrs_path, 'r', encoding='utf-8')
    attrs = fattrs.readlines()
    fattrs.close()
    label_attrs = {}
    for row in attrs:
        pair = row.strip().split(',')
        if pair[0] in unseen_labels[superclass]:
            label_attrs[pair[0]] = list(map(lambda x: float(x), pair[1].strip().split(' ')[1:-1]))
    attrs_entropy = calc_attrs_entropy(label_attrs)
    low_entropy_attrs = [i for i,ent in enumerate(attrs_entropy) if ent==0]
    print(low_entropy_attrs)

    fsave = open(low_entropy_attrs_path, 'w', encoding='utf-8')
    fsave.write(' '.join([str(attr) for attr in low_entropy_attrs])+'\n')
    fsave.close()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        superclass = sys.argv[1]
        calc_low_entropy_attrs(superclass)
    else:
        print('Param error')
        exit()
