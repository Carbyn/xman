#!/usr/bin/env python
"""Generate graph with features processed by PCA."""
import sys
from sklearn import svm

if len(sys.argv) != 2:
    print('naive_svc_linear.py SRC_PATH')
    exit(-1)

input_file = sys.argv[1]

print('1. Reading data...')
with open(input_file) as reader:
    data, category = [], []
    for line in reader:
        line = line.strip().split('\t')
        data.append(map(float, line[:-1]))
        category.append(int(line[-1]))

f1s = []
for mod in range(10):
    print('2.%d.1 Dividing data...' % mod)
    train_data, train_cat = [], []
    test_data, test_cat = [], []
    for i in range(len(data)):
        if i % 10 == mod:
            test_data.append(data[i])
            test_cat.append(category[i])
        else:
            train_data.append(data[i])
            train_cat.append(category[i])
    print('2.%d.2 Training...' % mod)
    clf = svm.SVC()
    clf.fit(train_data, train_cat)
    print('2.%d.3 Predicting...' % mod)
    predict = clf.predict(test_data)
    tp, hit = 0, 0
    for i in range(len(test_cat)):
        if predict[i] == test_cat[i]:
            hit += 1
            if test_cat[i] == 1:
                tp += 1
    print('tp=%d hit=%d predit=%d num=%d' % (tp, hit, sum(predict), sum(test_cat)))
    accuracy = 1.0 * (hit + 1e-12) / (len(predict) + 1e-12)
    precision = 1.0 * (tp + 1e-12) / (sum(predict) + 1e-12)
    recall = 1.0 * (tp + 1e-12) / (sum(test_cat) + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    print('A=%f P=%f R=%f F1=%f' % (accuracy, precision, recall, f1))
    f1s.append(f1)

print('3. Calculating average F1')
f1 = sum(f1s) / len(f1s)
print(f1)
