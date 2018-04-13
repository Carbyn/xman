#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import Counter
import numpy as np
import pickle
import json
import sys
import os

def load_labels(superclass):
    labels_path = '%s_test_labels.txt' % (superclass.lower())
    labels = {}
    with open(labels_path, 'r') as reader:
        for row in reader:
            pair = row.strip().split(' ')
            labels[pair[0]] = pair[1]
    return labels

def eval_cluster_result(superclass, images_cluster):
    labels = load_labels(superclass)
    eval_result = {}
    for idx, images in images_cluster.items():
        #print(idx, len(images))
        eval_result[idx] = Counter(map(lambda x: labels[x], images)).most_common(3)
    return eval_result

def main():
    if len(sys.argv) == 2:
        superclass = sys.argv[1]
    else:
        print('Param error')
        exit()

    cluster = 'AC'

    classNum = {'Animals': 10, 'Fruits': 10}

    features_path = 'features_%s.pickle' % (superclass)
    features_path_cluster = 'features_%s_cluster.pickle' % (superclass)

    fread = open(features_path, 'rb')
    fsave = open(features_path_cluster, 'wb')

    data_all = pickle.load(fread)
    features_all = data_all['features_all']
    labels_all = data_all['labels_all']
    images_all = data_all['images_all']

    test_labels_idxes = np.where(np.array(labels_all) == 'test')[0]
    test_features = list(np.array(features_all)[test_labels_idxes])
    test_images = list(np.array(images_all)[test_labels_idxes])

    if cluster == 'KMeans':
        clf = KMeans(n_clusters=classNum[superclass], max_iter=300)
        s = clf.fit(test_features)
    else:
        clf = AgglomerativeClustering(n_clusters=classNum[superclass], linkage='complete')
        cluster_labels = clf.fit_predict(test_features)
        ac_cluster_centers = {}
        for i in range(classNum[superclass]):
            cluster_labels_idxes = np.where(np.array(cluster_labels) == i)[0]
            sub_test_features = np.array(test_features)[cluster_labels_idxes]
            ac_cluster_centers[i] = list(np.mean(sub_test_features, axis=0))

    idx = 0
    images_cluster = {}
    for image in test_images:
        if cluster == 'KMeans':
            cluster_idx = clf.predict([test_features[idx]])
            cluster_idx = cluster_idx[0]
            feature_cluster = clf.cluster_centers_[cluster_idx]
        else :
            cluster_idx = cluster_labels[idx]
            feature_cluster = ac_cluster_centers[cluster_idx]
        features_all[test_labels_idxes[idx]] = feature_cluster
        if str(cluster_idx) not in images_cluster.keys():
            images_cluster[str(cluster_idx)] = []
        images_cluster[str(cluster_idx)].append(image)
        idx += 1

    fimages = open('images_cluster.json', 'w')
    fimages.write(json.dumps(images_cluster))
    fimages.close()

    feval = open('images_cluster_eval.json', 'w')
    feval.write(json.dumps(eval_cluster_result(superclass, images_cluster)))
    feval.close()

    data_all = {'features_all':features_all, 'labels_all':labels_all,
                'images_all':images_all}

    pickle.dump(data_all, fsave)

    fread.close()
    fsave.close()


if __name__ == "__main__":
    main()
