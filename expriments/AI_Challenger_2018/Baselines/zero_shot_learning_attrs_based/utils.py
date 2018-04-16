#!/usr/bin/env python
# coding=utf-8
import numpy as np

def calc_entropy(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def remove_low_entropy_attrs(label_attrs, entropy_thr=0.0):
    label_attrs_removed = {}
    valid_attr_idxes = []
    attrs = np.array(list(label_attrs.values()))
    print(attrs.shape[1])
    for i in range(attrs.shape[1]):
        entropy = calc_entropy(attrs[:, i])
        print(i, entropy)
        if entropy >= entropy_thr:
            valid_attr_idxes.append(i)
    for label, attrs in label_attrs.items():
        label_attrs_removed[label] = list(np.array(attrs)[valid_attr_idxes])
    return label_attrs_removed, valid_attr_idxes

def remove_non_visible_attrs(label_attrs, superclass):
    with open('attr_valid_idxes_%s.txt' % (superclass.lower()), 'r') as reader:
        label_attrs_removed = {}
        valid_attr_idxes = [int(row.strip()) - 1 for row in reader.readlines()]
        for label, attrs in label_attrs.items():
            label_attrs_removed[label] = list(np.array(attrs)[valid_attr_idxes])
        return label_attrs_removed, valid_attr_idxes
def attrs_reduce(class_attrs_path, superclass, entropy_thr = 0):
    fattrs = open(class_attrs_path, 'r', encoding='utf-8')
    attrs = fattrs.readlines()
    fattrs.close()
    label_attrs = {}
    for row in attrs:
        pair = row.strip().split(',')
        label_attrs[pair[0]] = list(map(lambda x: float(x), pair[1].strip().split(' ')[1:-1]))
    label_attrs, label_attrs_idxes = remove_non_visible_attrs(label_attrs, superclass)
    label_attrs, label_attrs_idxes = remove_low_entropy_attrs(label_attrs, entropy_thr)
    return label_attrs
