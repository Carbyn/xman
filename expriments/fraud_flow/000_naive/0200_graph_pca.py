#!/usr/bin/env python
"""Generate graph with features processed by PCA."""
import sys
import numpy as np
import sklearn.decomposition as deco
from matplotlib import pyplot as plt

if len(sys.argv) != 3:
    print('graph_pca.py SRC_PATH DST_PATH')
    exit(-1)

input_file = sys.argv[1]
output_file = sys.argv[2]

print('1. Reading data...')
with open(input_file) as reader:
    data, category = [], []
    for line in reader:
        line = line.strip().split('\t')
        data.append(map(float, line[:-1]))
        category.append(line[-1])

print('2. Calculating PCA...')
data = (data - np.mean(data, 0)) / np.std(data, 0)
pca = deco.PCA(2)
reduced = pca.fit(data).transform(data)

print('3. Drawing scatter graph...')
normal_x, normal_y = [], []
spider_x, spider_y = [], []
for i in range(len(data)):
    if category[i] == '0':
        normal_x.append(reduced[i][0])
        normal_y.append(reduced[i][1])
    else:
        spider_x.append(reduced[i][0])
        spider_y.append(reduced[i][1])
axes = plt.subplot(111)
axes.scatter(normal_x, normal_y, s=1, c='green', alpha=0.5)
axes.scatter(spider_x, spider_y, s=1, c='red', alpha=0.5)
plt.savefig(output_file, dpi=600)
plt.show()
