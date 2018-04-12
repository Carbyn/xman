#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
import pickle
import json
import sys
import os


def main():
    if len(sys.argv) == 2:
        superclass = sys.argv[1]
    else:
        print('Param error')
        exit()

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
    #print(len(test_labels_idxes), len(test_features), len(test_images))

    clf = KMeans(n_clusters=classNum[superclass], max_iter=300)
    s = clf.fit(features_all)

    idx = 0
    images_cluster = {}
    for image in test_images:
        cluster_idx = clf.predict([test_features[idx]])
        cluster_idx = cluster_idx[0]
        feature_cluster = clf.cluster_centers_[cluster_idx]
        #print(len(features_all), len(test_labels_idxes), idx)
        features_all[test_labels_idxes[idx]] = feature_cluster
        if str(cluster_idx) not in images_cluster.keys():
            images_cluster[str(cluster_idx)] = []
        images_cluster[str(cluster_idx)].append(image)
        idx += 1

    fimages = open('images_cluster.json', 'w')
    fimages.write(json.dumps(images_cluster))
    fimages.close()

    data_all = {'features_all':features_all, 'labels_all':labels_all,
                'images_all':images_all}

    pickle.dump(data_all, fsave)

    fread.close()
    fsave.close()


if __name__ == "__main__":
    main()
