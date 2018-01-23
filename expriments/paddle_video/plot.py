import json
import numpy as np
import matplotlib  
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from reader import load_json
from collections import Counter


def get_gt_data(video):
    file_path = 'data/meta.json' 
    json_data = load_json(file_path)
    database = json_data['database']
    data_x = []
    data_y = []
    for i in database[video]['annotations']: 
        st, ed = i['segment']
        data_x.append(st)
        data_y.append(0)
        data_x.append(st)
        data_y.append(1)
        data_x.append(ed)
        data_y.append(1)
        data_x.append(ed)
        data_y.append(0)
    return data_x, data_y


def get_pr_data(video, dataset):
    file_path = 'res/' + dataset + '.json' 
    json_data = load_json(file_path)
    database = json_data['results']
    data_x = []
    data_y = []
    for i in database[video]: 
        st, ed = i['segment']
        data_x.append(st)
        data_y.append(0)
        data_x.append(st)
        data_y.append(i['score'])
        data_x.append(ed)
        data_y.append(i['score'])
        data_x.append(ed)
        data_y.append(0)
    return data_x, data_y


def get_bd_data(video, dataset):
    file_path = 'res/' + dataset + '.json' 
    json_data = load_json(file_path)
    database = json_data['results']
    data_x = []
    data_y = []
    for i in database[video]: 
        st, ed = i['segment']
        data_x.append(st)
        data_y.append(0)
        data_x.append(st)
        data_y.append(1)
        data_x.append(ed)
        data_y.append(1)
        data_x.append(ed)
        data_y.append(0)
    return data_x, data_y


def plot(dataset):
    file_path = 'data/meta.json' 
    json_data = load_json("data/meta.json")
    database = json_data['database']
    data_x = []
    data_y = []
    for video in database.keys():
        if database[video]['subset'] != 'validation':
            continue
        fig, ax = plt.subplots()
        gt_x, gt_y = get_gt_data(video)
        pr_x, pr_y = get_pr_data(video, dataset)
        bd_x, bd_y = get_bd_data(video, dataset)
        line1, = ax.plot(gt_x, gt_y)
        line2, = ax.plot(pr_x, pr_y, '--', linewidth=1)
        plt.savefig('image/' + dataset + '/' + video + '.jpg')


def stat():
    file_path = 'data/meta.json' 
    json_data = load_json(file_path)
    database = json_data['database']
    data = []
    for video in database.keys():
        vd = []
        if database[video]['subset'] != 'validation':
            continue
        for i in database[video]['annotations']: 
            st, ed = i['segment']
            vd.append(int((ed - st) / 10.0) * 10)
        mlen = int(np.mean(vd) / 10) * 10
        data.append(mlen)
    print Counter(data).keys()


plot('validation')
plot('validation_refine')
#stat()
